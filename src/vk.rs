// src/vk.rs
use anyhow::{Context, Result};
use ash::{vk, Entry};
use bytemuck::{Pod, Zeroable};
use std::{ffi::CString, mem::size_of};
use crate::backend::MatMul;

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
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
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

        let priorities = [1.0f32];
        let qinfo = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(qf_index)
            .queue_priorities(&priorities)
            .build()];
        let dinfo = vk::DeviceCreateInfo::builder().queue_create_infos(&qinfo);
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
        let pl_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[desc_set_layout])
            .push_constant_ranges(&[pc_range]);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pl_info, None)? };

        // Shader module
        // NOTE: SPIR-V must be 4-byte aligned; include_bytes! is fine and we cast safely.
        let spirv: &[u8] = include_bytes!("../shaders/matmul_f32_t16x16.spv");
        assert!(spirv.len() % 4 == 0, "SPIR-V must be 4-byte aligned");
        let words: &[u32] = bytemuck::cast_slice(spirv);
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

        // Command pool
        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(qf_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { device.create_command_pool(&pool_info, None)? };

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
            desc_set_layout, desc_pool,
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

impl MatMul for VkBackend {
    fn matmul(&mut self, m: usize, n: usize, k: usize,
              a: &[f32], b: &[f32], c: &mut [f32]) -> Result<()> {
        let bytes_a = (a.len() * 4) as u64;
        let bytes_b = (b.len() * 4) as u64;
        let bytes_c = (c.len() * 4) as u64;

        let (buf_a, mem_a) = self.alloc_host_buffer(bytes_a)?;
        let (buf_b, mem_b) = self.alloc_host_buffer(bytes_b)?;
        let (buf_c, mem_c) = self.alloc_host_buffer(bytes_c)?;
        self.write_buffer_pod(mem_a, a)?;
        self.write_buffer_pod(mem_b, b)?;

        // Descriptor set
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&[self.desc_set_layout]);
        let desc_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? }[0];

        let descs = [
            vk::DescriptorBufferInfo { buffer: buf_a, offset: 0, range: bytes_a },
            vk::DescriptorBufferInfo { buffer: buf_b, offset: 0, range: bytes_b },
            vk::DescriptorBufferInfo { buffer: buf_c, offset: 0, range: bytes_c },
        ];
        let writes = [
            vk::WriteDescriptorSet {
                dst_set: desc_set, dst_binding: 0, descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[0], ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set, dst_binding: 1, descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[1], ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: desc_set, dst_binding: 2, descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &descs[2], ..Default::default()
            },
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        // Command buffer
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
            let pc = PushConsts {
                M: m as u32, N: n as u32, K: k as u32,
                lda: k as u32, ldb: n as u32, ldc: n as u32,
            };
            self.device.cmd_push_constants(
                cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE,
                0, bytemuck::bytes_of(&pc));

            let gx = ((n as u32) + 15) / 16;
            let gy = ((m as u32) + 15) / 16;
            self.device.cmd_dispatch(cmd, gx, gy, 1);
            self.device.end_command_buffer(cmd)?;
        }

        // Submit + wait
        let submit = vk::SubmitInfo::builder().command_buffers(&[cmd]);
        unsafe {
            self.device.queue_submit(self.queue, &[*submit], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        // Read back C
        unsafe {
            let ptr = self.device.map_memory(mem_c, 0, bytes_c, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(ptr as *const f32, c.as_mut_ptr(), c.len());
            self.device.unmap_memory(mem_c);
        }

        // Cleanup temporaries
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            // freeing descriptor sets is optional; pool reset/destroy will reclaim. We'll free now:
            self.device.free_descriptor_sets(self.desc_pool, &[desc_set]).ok();
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
