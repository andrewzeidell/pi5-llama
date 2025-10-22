// src/vk.rs
use anyhow::{Context, Result};
use ash::{vk, Entry};
use bytemuck::{Pod, Zeroable};
use std::{ffi::CString, mem::size_of};
use crate::backend::{MatMul, Softmax};

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PushConsts {
    M: u32, N: u32, K: u32,
    lda: u32, ldb: u32, ldc: u32,
}

pub struct VkBackend {
    entry: Entry,
    instance: ash::Instance,
    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    qf_index: u32,
    //use_f16: bool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    query_pool: vk::QueryPool,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,

    pipeline_fused: Option<vk::Pipeline>,
    pipeline_layout_fused: Option<vk::PipelineLayout>,
    desc_set_layout_fused: Option<vk::DescriptorSetLayout>,
    desc_pool_fused: Option<vk::DescriptorPool>,

    buf_q: Option<vk::Buffer>,
    mem_q: Option<vk::DeviceMemory>,
    buf_k: Option<vk::Buffer>,
    mem_k: Option<vk::DeviceMemory>,
    buf_v: Option<vk::Buffer>,
    mem_v: Option<vk::DeviceMemory>,
    buf_o: Option<vk::Buffer>,
    mem_o: Option<vk::DeviceMemory>,
}

impl VkBackend {
    pub fn init_attention_fused(&mut self, m: usize, n: usize, d: usize, dv: usize) -> anyhow::Result<()> {
        use std::ffi::CString;
        use std::mem::size_of;

        // ─── 1. Descriptor set layout ───────────────────────────────
        let mk_binding = |binding: u32| vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        };
        let bindings = [mk_binding(0), mk_binding(1), mk_binding(2), mk_binding(3)];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let dsl = unsafe { self.device.create_descriptor_set_layout(&dsl_info, None)? };

        // ─── 2. Pipeline layout + push constants ─────────────────────
        #[repr(C)]
        struct PC { m: u32, n: u32, d: u32, dv: u32 }
        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<PC>() as u32,
        };
        let layouts = [dsl];
        let pc_ranges = [pc_range];
        let pl_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&pc_ranges);
        let pl = unsafe { self.device.create_pipeline_layout(&pl_info, None)? };

        // ─── 3. Shader module + pipeline ─────────────────────────────
        let spirv_bytes: &[u8] = include_bytes!("../shaders/attn_fused.spv");
        let (_, words, _) = unsafe { spirv_bytes.align_to::<u32>() };
        let sm_info = vk::ShaderModuleCreateInfo::builder().code(words);
        let shader_module = unsafe { self.device.create_shader_module(&sm_info, None)? };
        let entry_point = CString::new("main")?;
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);
        let cp_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(pl);
        let pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[*cp_info], None)
        }.map_err(|e| anyhow::anyhow!("pipeline create failed: {:?}", e))?[0];
        unsafe { self.device.destroy_shader_module(shader_module, None) };

        // ─── 4. Descriptor pool ─────────────────────────────────────
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 4,
        }];
        let dp_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let dp = unsafe { self.device.create_descriptor_pool(&dp_info, None)? };

        // ─── 5. Persistent buffers ──────────────────────────────────
        let alloc_buf = |sz| -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
            let buf_info = vk::BufferCreateInfo::builder()
                .size(sz)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = unsafe { self.device.create_buffer(&buf_info, None)? };
            let mem_req = unsafe { self.device.get_buffer_memory_requirements(buf) };
            let mem_props = unsafe { self.instance.get_physical_device_memory_properties(self.phys) };
            let mem_type = (0..mem_props.memory_type_count)
                .find(|&i| {
                    let mt = mem_props.memory_types[i as usize];
                    (mem_req.memory_type_bits & (1 << i)) != 0 &&
                    mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
                })
                .context("no HOST_VISIBLE|HOST_COHERENT memory type")?;
            let alloc = vk::MemoryAllocateInfo {
                allocation_size: mem_req.size,
                memory_type_index: mem_type,
                ..Default::default()
            };
            let mem = unsafe { self.device.allocate_memory(&alloc, None)? };
            unsafe { self.device.bind_buffer_memory(buf, mem, 0)? };
            Ok((buf, mem))
        };

        let bytes_q = (m * d * 4) as u64;
        let bytes_k = (n * d * 4) as u64;
        let bytes_v = (n * dv * 4) as u64;
        let bytes_o = (m * dv * 4) as u64;

        let (buf_q, mem_q) = alloc_buf(bytes_q)?;
        let (buf_k, mem_k) = alloc_buf(bytes_k)?;
        let (buf_v, mem_v) = alloc_buf(bytes_v)?;
        let (buf_o, mem_o) = alloc_buf(bytes_o)?;

        // ─── Save ───────────────────────────────────────────────────
        self.pipeline_fused = Some(pipeline);
        self.pipeline_layout_fused = Some(pl);
        self.desc_set_layout_fused = Some(dsl);
        self.desc_pool_fused = Some(dp);
        self.buf_q = Some(buf_q); self.mem_q = Some(mem_q);
        self.buf_k = Some(buf_k); self.mem_k = Some(mem_k);
        self.buf_v = Some(buf_v); self.mem_v = Some(mem_v);
        self.buf_o = Some(buf_o); self.mem_o = Some(mem_o);

        Ok(())
    }

    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load().context("load Vulkan entry")? };
        let app_name = CString::new("pi5-llama")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_2);
        let inst_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&inst_info, None)? };

        // Choose device + queue
        let phys = unsafe { instance.enumerate_physical_devices()? }
            .into_iter().next().context("no physical device")?;
        let qf_index = unsafe { instance.get_physical_device_queue_family_properties(phys) }
            .iter().enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .context("no compute queue")?;

        let priorities = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(qf_index)
            .queue_priorities(&priorities)
            .build()];
        let dinfo = vk::DeviceCreateInfo::builder().queue_create_infos(&qinfo);
        let device = unsafe { instance.create_device(phys, &dinfo, None)? };
        let queue = unsafe { device.get_device_queue(qf_index, 0) };

        // Descriptor + pipeline layout for matmul
        let mk_binding = |binding: u32| vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        };
        let bindings = [mk_binding(0), mk_binding(1), mk_binding(2)];
        let desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let desc_set_layout = unsafe { device.create_descriptor_set_layout(&desc_set_layout_info, None)? };

        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<PushConsts>() as u32,
        };
        let layouts = [desc_set_layout];
        let pc_ranges = [pc_range];
        let pl_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&pc_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_info, None)? };

        // Matmul shader pipeline (load SPIR-V at runtime to avoid alignment issues)
        use std::fs::File;
        use std::io::Read;
        let mut f = File::open("shaders/matmul_f32_t16x16.spv")
            .context("open shaders/matmul_f32_t16x16.spv")?;
        let mut spirv = Vec::new();
        f.read_to_end(&mut spirv)?;
        assert!(spirv.len() % 4 == 0, "SPIR-V must be multiple of 4 bytes");
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(spirv.as_ptr() as *const u32, spirv.len() / 4)
        };
        let sm_info = vk::ShaderModuleCreateInfo::builder().code(words);
        let shader_module = unsafe { device.create_shader_module(&sm_info, None)? };

        let entry_point = CString::new("main")?;
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);
        let cp_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[*cp_info], None)
        }.map_err(|e| anyhow::anyhow!("pipeline create failed: {:?}", e))?[0];
        unsafe { device.destroy_shader_module(shader_module, None) };

        // Command pool + query pool
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(qf_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        let qinfo = vk::QueryPoolCreateInfo {
            s_type: vk::StructureType::QUERY_POOL_CREATE_INFO,
            query_type: vk::QueryType::TIMESTAMP,
            query_count: 2,
            ..Default::default()
        };
        let query_pool = unsafe { device.create_query_pool(&qinfo, None)? };

        // Descriptor pool for matmul
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let dp_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let desc_pool = unsafe { device.create_descriptor_pool(&dp_info, None)? };

        Ok(Self {
            entry, instance, phys, device, queue, qf_index,
            pipeline_layout, pipeline, cmd_pool,
            desc_set_layout, desc_pool, query_pool, 
        
            // new persistent resources (initialize to None)
            pipeline_fused: None,
            pipeline_layout_fused: None,
            desc_set_layout_fused: None,
            desc_pool_fused: None,
            buf_q: None,
            mem_q: None,
            buf_k: None,
            mem_k: None,
            buf_v: None,
            mem_v: None,
            buf_o: None,
            mem_o: None,
        })
    }

    fn alloc_host_buffer(&self, size: vk::DeviceSize) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buf_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buf_info, None)? };
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe { self.instance.get_physical_device_memory_properties(self.phys) };
        let mem_type = (0..mem_props.memory_type_count)
            .find(|&i| {
                let mt = mem_props.memory_types[i as usize];
                (mem_req.memory_type_bits & (1 << i)) != 0
                    && mt.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .context("no HOST_VISIBLE|HOST_COHERENT memory type")?;
        let alloc = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: mem_type,
            ..Default::default()
        };
        let memory = unsafe { self.device.allocate_memory(&alloc, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        Ok((buffer, memory))
    }

    fn write_buffer_bytes(&self, mem: vk::DeviceMemory, src: &[u8]) -> Result<()> {
        unsafe {
            let ptr = self.device.map_memory(mem, 0, src.len() as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr as *mut u8, src.len());
            self.device.unmap_memory(mem);
        }
        Ok(())
    }

    fn write_buffer_pod<T: Pod>(&self, mem: vk::DeviceMemory, data: &[T]) -> Result<()> {
        self.write_buffer_bytes(mem, bytemuck::cast_slice(data))
    }
    pub fn attention_fused(
        &mut self,
        m: usize, n: usize, d: usize, dv: usize,
        q: &[f32], k: &[f32], v: &[f32], o: &mut [f32],
    ) -> anyhow::Result<()> {
        let dev = &self.device;
    
        // Upload Q, K, V
        unsafe {
            let map = |mem: vk::DeviceMemory, data: &[f32]| -> anyhow::Result<()> {
                let ptr = dev.map_memory(mem, 0, (data.len()*4) as u64, vk::MemoryMapFlags::empty())?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
                dev.unmap_memory(mem);
                Ok(())
            };
            map(self.mem_q.unwrap(), q)?;
            map(self.mem_k.unwrap(), k)?;
            map(self.mem_v.unwrap(), v)?;
        }
    
        // Descriptor set
        let set_layouts = [self.desc_set_layout_fused.unwrap()];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.desc_pool_fused.unwrap())
            .set_layouts(&set_layouts);
        let desc_set = unsafe { dev.allocate_descriptor_sets(&alloc_info)? }[0];
    
        let descs = [
            vk::DescriptorBufferInfo { buffer: self.buf_q.unwrap(), offset: 0, range: (q.len()*4) as u64 },
            vk::DescriptorBufferInfo { buffer: self.buf_k.unwrap(), offset: 0, range: (k.len()*4) as u64 },
            vk::DescriptorBufferInfo { buffer: self.buf_v.unwrap(), offset: 0, range: (v.len()*4) as u64 },
            vk::DescriptorBufferInfo { buffer: self.buf_o.unwrap(), offset: 0, range: (o.len()*4) as u64 },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..4).map(|i| vk::WriteDescriptorSet {
            dst_set: desc_set,
            dst_binding: i,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &descs[i as usize],
            ..Default::default()
        }).collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };
    
        // Command buffer record
        let alloc = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { dev.allocate_command_buffers(&alloc)? }[0];
        let begin = vk::CommandBufferBeginInfo::builder();
        unsafe {
            dev.begin_command_buffer(cmd, &begin)?;
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_fused.unwrap());
            dev.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout_fused.unwrap(), 0, &[desc_set], &[]);
            let pc = [m as u32, n as u32, d as u32, dv as u32];
            dev.cmd_push_constants(
                cmd, self.pipeline_layout_fused.unwrap(), vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::cast_slice(&pc));
            dev.cmd_dispatch(cmd, m as u32, 1, 1);
            dev.end_command_buffer(cmd)?;
        }
    
        // Submit and wait
       let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);
        unsafe {
            dev.queue_submit(self.queue, &[*submit_info], vk::Fence::null())?;
            dev.queue_wait_idle(self.queue)?;
        }
    
        // Read back O
        unsafe {
            let ptr = dev.map_memory(self.mem_o.unwrap(), 0, (o.len()*4) as u64, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, o.as_mut_ptr(), o.len());
            dev.unmap_memory(self.mem_o.unwrap());
        }
    
        // Clean up descriptor set (pool is reused)
        unsafe { dev.free_command_buffers(self.cmd_pool, &[cmd]) };
        Ok(())
    }
}

// -----------------------------------------------------
// MATMUL IMPLEMENTATION
// -----------------------------------------------------
impl MatMul for VkBackend {
    fn matmul(&mut self, m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
        let bytes_a = (a.len() * 4) as u64;
        let bytes_b = (b.len() * 4) as u64;
        let bytes_c = (c.len() * 4) as u64;
        let (buf_a, mem_a) = self.alloc_host_buffer(bytes_a)?;
        let (buf_b, mem_b) = self.alloc_host_buffer(bytes_b)?;
        let (buf_c, mem_c) = self.alloc_host_buffer(bytes_c)?;
        self.write_buffer_pod(mem_a, a)?;
        self.write_buffer_pod(mem_b, b)?;

        let set_layouts = [self.desc_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&set_layouts);
        let desc_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? }[0];

        let descs = [
            vk::DescriptorBufferInfo { buffer: buf_a, offset: 0, range: bytes_a },
            vk::DescriptorBufferInfo { buffer: buf_b, offset: 0, range: bytes_b },
            vk::DescriptorBufferInfo { buffer: buf_c, offset: 0, range: bytes_c },
        ];
        let writes = [
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[0],
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[1],
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[2],
                ..Default::default()
            },
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let alloc = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.device.allocate_command_buffers(&alloc)? }[0];

        let begin = vk::CommandBufferBeginInfo::builder();
        unsafe {
            self.device.begin_command_buffer(cmd, &begin)?;
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[desc_set], &[]);
            let pc = PushConsts { M: m as u32, N: n as u32, K: k as u32, lda: k as u32, ldb: n as u32, ldc: n as u32 };
            self.device.cmd_push_constants(
                cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc));
            let gx = ((n as u32) + 15) / 16;
            let gy = ((m as u32) + 15) / 16;
            self.device.cmd_dispatch(cmd, gx, gy, 1);
            self.device.end_command_buffer(cmd)?;
        }

        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);
        unsafe {
            self.device.queue_submit(self.queue, &[*submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            let ptr = self.device.map_memory(mem_c, 0, bytes_c, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, c.as_mut_ptr(), c.len());
            self.device.unmap_memory(mem_c);
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device.destroy_buffer(buf_a, None);
            self.device.free_memory(mem_a, None);
            self.device.destroy_buffer(buf_b, None);
            self.device.free_memory(mem_b, None);
            self.device.destroy_buffer(buf_c, None);
            self.device.free_memory(mem_c, None);
        }
        Ok(())
    }
}

// -----------------------------------------------------
// SOFTMAX IMPLEMENTATION (fixed descriptor+pipeline order)
// -----------------------------------------------------
impl Softmax for VkBackend {
    fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        assert_eq!(x.len(), rows * cols, "softmax_rows: x len != rows*cols");

        // Load shader
        let mut file = File::open("shaders/softmax_rows.spv")?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        assert!(bytes.len() % 4 == 0);
        let words = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };
        let sm_info = vk::ShaderModuleCreateInfo::builder().code(words);
        let shader_module = unsafe { self.device.create_shader_module(&sm_info, None)? };

        // Descriptor set layout FIRST
        let binding0 = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        };
        let bindings = [binding0];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let dsl = unsafe { self.device.create_descriptor_set_layout(&dsl_info, None)? };

        // Pipeline layout includes push constants + set layout
        let pc_range = vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 8 };
        let pc_ranges = [pc_range];
        let set_layouts = [dsl];
        let pl_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_ranges);
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&pl_info, None)? };

        // Pipeline
        let entry_point = CString::new("main")?;
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);
        let cp_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[*cp_info], None)
        }.map_err(|e| anyhow::anyhow!("softmax pipeline create failed: {:?}", e))?[0];
        unsafe { self.device.destroy_shader_module(shader_module, None) };

        // Buffer
        let bytes_x = (x.len() * 4) as u64;
        let (buf_x, mem_x) = self.alloc_host_buffer(bytes_x)?;
        self.write_buffer_pod(mem_x, x)?;

        // Descriptor pool + set
        let pool_sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 1 }];
        let dp_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(1);
        let dpool = unsafe { self.device.create_descriptor_pool(&dp_info, None)? };

        let set_layouts = [dsl];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(dpool)
            .set_layouts(&set_layouts);
        let dset = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? }[0];

        let buf_info = vk::DescriptorBufferInfo { buffer: buf_x, offset: 0, range: bytes_x };
        let write = vk::WriteDescriptorSet {
            dst_set: dset,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buf_info,
            ..Default::default()
        };
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        // Command buffer
        let alloc = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { self.device.allocate_command_buffers(&alloc)? }[0];

        let begin = vk::CommandBufferBeginInfo::builder();
        let pc_data = [rows as u32, cols as u32];
        unsafe {
            self.device.begin_command_buffer(cmd, &begin)?;
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[dset], &[]);
            self.device.cmd_push_constants(
                cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&pc_data));
            // One workgroup per row (shader uses gl_WorkGroupID.x as row index)
            self.device.cmd_dispatch(cmd, rows as u32, 1, 1);
            self.device.end_command_buffer(cmd)?;
        }

        // Submit + wait
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);
        unsafe {
            self.device.queue_submit(self.queue, &[*submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        // Read back
        unsafe {
            let ptr = self.device.map_memory(mem_x, 0, bytes_x, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, x.as_mut_ptr(), x.len());
            self.device.unmap_memory(mem_x);
        }

        // Cleanup
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device.destroy_descriptor_pool(dpool, None);
            self.device.destroy_descriptor_set_layout(dsl, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_buffer(buf_x, None);
            self.device.free_memory(mem_x, None);
        }

        Ok(())
    }
}
