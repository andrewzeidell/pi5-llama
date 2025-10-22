// src/vk.rs
use anyhow::{Context, Result};
use ash::{vk, Entry};
use half::f16;
use bytemuck::{Pod, Zeroable};
use std::{ffi::CString, mem::size_of};
use crate::backend::Softmax;

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
    use_f16: bool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    query_pool: vk::QueryPool,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
}

impl VkBackend {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load().context("load Vulkan entry")? };
        let app_name = CString::new("pi5-llama")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .api_version(vk::API_VERSION_1_2);

        let inst_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&inst_info, None)? };

        // Choose a device and a compute queue
        let phys = unsafe { instance.enumerate_physical_devices()? }
            .into_iter().next().context("no physical device")?;
        let qf_index = unsafe { instance.get_physical_device_queue_family_properties(phys) }
            .iter().enumerate()
            .find(|(_, q)| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .context("no compute queue")?;
        // --- Query features we need for f16 SSBO + arithmetic ---
        let mut f16i8 = vk::PhysicalDeviceFloat16Int8FeaturesKHR::default();
        let mut s16 = vk::PhysicalDevice16BitStorageFeaturesKHR::default();
        let mut feats2 = vk::PhysicalDeviceFeatures2::default();
        feats2.p_next = (&mut f16i8 as *mut _) as *mut std::ffi::c_void;
        f16i8.p_next = (&mut s16 as *mut _) as *mut std::ffi::c_void;
        
        unsafe { instance.get_physical_device_features2(phys, &mut feats2); }
        
        let use_f16 = f16i8.shader_float16 == vk::TRUE && s16.storage_buffer16_bit_access == vk::TRUE;
        println!(
            "Device reports: shaderFloat16={} storageBuffer16BitAccess={}",
            if f16i8.shader_float16 == vk::TRUE { "YES" } else { "no" },
            if s16.storage_buffer16_bit_access == vk::TRUE { "YES" } else { "no" }
        );
        println!(
            "FP16 pipeline {}",
            if use_f16 { "ENABLED ✅" } else { "NOT SUPPORTED ⚠️ (falling back to f32)" }
        );



        let priorities = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(qf_index)
            .queue_priorities(&priorities)
            .build()];
        
        // enable features via pNext chain
        let mut f16i8_enable = vk::PhysicalDeviceFloat16Int8FeaturesKHR {
            shader_float16: vk::TRUE,
            ..Default::default()
        };
        let mut s16_enable = vk::PhysicalDevice16BitStorageFeaturesKHR {
            storage_buffer16_bit_access: vk::TRUE,
            ..Default::default()
        };
        let mut feats2_enable = vk::PhysicalDeviceFeatures2::default();
        feats2_enable.p_next = (&mut f16i8_enable as *mut _) as *mut std::ffi::c_void;
        f16i8_enable.p_next = (&mut s16_enable as *mut _) as *mut std::ffi::c_void;
        
        if !use_f16 {
            f16i8_enable.shader_float16 = vk::FALSE;
            s16_enable.storage_buffer16_bit_access = vk::FALSE;
        }
        
        let dinfo = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&qinfo)
            .push_next(&mut feats2_enable); // this .push_next is fine (builder-level)

        
        let device = unsafe { instance.create_device(phys, &dinfo, None)? };
        let queue = unsafe { device.get_device_queue(qf_index, 0) };

        // Descriptor set layout: A,B,C as STORAGE_BUFFER
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

        // Pipeline layout (push constants)
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

        // Shader module
        // NOTE: SPIR-V must be 4-byte aligned; include_bytes! is fine and we cast safely.
        use std::fs::File;
        use std::io::Read;
        
        let shader_path = if use_f16 {
            "shaders/matmul_f16_t16x16.spv"
        } else {
            "shaders/matmul_f32_t16x16.spv"
        };
        
        let mut file = File::open(shader_path)
            .with_context(|| format!("failed to open shader file: {}", shader_path))?;
        let mut spirv_bytes = Vec::new();
        file.read_to_end(&mut spirv_bytes)?;
        assert!(spirv_bytes.len() % 4 == 0, "SPIR-V size not multiple of 4 bytes");
        
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(spirv_bytes.as_ptr() as *const u32, spirv_bytes.len() / 4)
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
        
        // Save pipeline on self
        // (… then in the Ok(Self { … }) above, set `pipeline` to this `pipeline`)


        // Command pool
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(qf_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };

        // Query pool
        let qp_info = vk::QueryPoolCreateInfo {
            query_type: vk::QueryType::TIMESTAMP,
            query_count: 2,
            ..Default::default()
        };
        let query_pool = unsafe { device.create_query_pool(&qp_info, None)? };

        // Descriptor pool
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
            query_pool, desc_set_layout, desc_pool, 
            use_f16,
        })
    }


    pub fn softmax_rows(&mut self, rows: usize, cols: usize, x: &mut [f32]) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        // --- Load the SPIR-V shader ---
        let mut file = File::open("shaders/softmax_rows.spv")?;
        let mut spirv_bytes = Vec::new();
        file.read_to_end(&mut spirv_bytes)?;
        let words: &[u32] = unsafe {
            std::slice::from_raw_parts(spirv_bytes.as_ptr() as *const u32, spirv_bytes.len() / 4)
        };
        let sm_info = vk::ShaderModuleCreateInfo::builder().code(words);
        let shader_module = unsafe { self.device.create_shader_module(&sm_info, None)? };

        // --- Create pipeline ---
        let entry_point = std::ffi::CString::new("main")?;
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<[u32; 2]>() as u32,
        };
        let pc_ranges = [pc_range];
        let pl_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&pc_ranges);
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&pl_info, None)? };
        let cp_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            self.device.create_compute_pipelines(vk::PipelineCache::null(), &[*cp_info], None)
        }.map_err(|e| anyhow::anyhow!("softmax pipeline create failed: {:?}", e))?[0];
        unsafe { self.device.destroy_shader_module(shader_module, None) };

        // --- Upload data ---
        let bytes_x = (x.len() * 4) as u64;
        let (buf_x, mem_x) = self.alloc_host_buffer(bytes_x)?;
        self.write_buffer_pod(mem_x, x)?;

        // Descriptor set
        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
        }];
        let desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let desc_set_layout = unsafe { self.device.create_descriptor_set_layout(&desc_set_layout_info, None)? };
        let desc_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        }];
        let dp_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&desc_pool_sizes)
            .max_sets(1);
        let desc_pool = unsafe { self.device.create_descriptor_pool(&dp_info, None)? };
        let set_layouts = [desc_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let desc_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? }[0];

        let buf_info = vk::DescriptorBufferInfo { buffer: buf_x, offset: 0, range: bytes_x };
        let write = vk::WriteDescriptorSet {
            dst_set: desc_set, dst_binding: 0, descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &buf_info, ..Default::default()
        };
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        // --- Command buffer ---
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
                cmd, vk::PipelineBindPoint::COMPUTE, pipeline_layout, 0, &[desc_set], &[]);
            self.device.cmd_push_constants(
                cmd, pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&pc_data));
            self.device.cmd_dispatch(cmd, rows as u32, 1, 1);
            self.device.end_command_buffer(cmd)?;
        }

        // --- Submit + wait ---
        let cmd_bufs = [cmd];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_bufs);
        unsafe {
            self.device.queue_submit(self.queue, &[*submit_info], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        // --- Read back result ---
        unsafe {
            let ptr = self.device.map_memory(mem_x, 0, bytes_x, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, x.as_mut_ptr(), x.len());
            self.device.unmap_memory(mem_x);
        }

        // --- Cleanup ---
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device.destroy_buffer(buf_x, None);
            self.device.free_memory(mem_x, None);
            self.device.destroy_descriptor_pool(desc_pool, None);
            self.device.destroy_descriptor_set_layout(desc_set_layout, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_pipeline(pipeline, None);
        }

        Ok(())
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
                (mem_req.memory_type_bits & (1 << i)) != 0 &&
                mt.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
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
}




