#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * ggml_backend_buffer_type_context_t;

    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft);
        // max buffer size that can be allocated
        size_t                (*get_max_size)    (ggml_backend_buffer_type_t buft);
        // data size needed to allocate the tensor, including padding
        size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // check if tensor data is in host memory
        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * ggml_backend_buffer_context_t;

    struct ggml_backend_buffer_i {
        const char * (*get_name)      (ggml_backend_buffer_t buffer);
        void         (*free_buffer)   (ggml_backend_buffer_t buffer);
        void *       (*get_base)      (ggml_backend_buffer_t buffer);
        void         (*init_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        void         (*memset_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)    (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)    (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*cpy_tensor)    (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*clear)         (ggml_backend_buffer_t buffer, uint8_t value);
        void         (*reset)         (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        ggml_backend_buffer_context_t context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t      buft,
            struct ggml_backend_buffer_i           iface,
                   ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use ggml_backend_tensor_copy instead
    bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend
    //

    typedef void * ggml_backend_context_t;

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // buffer allocation
        ggml_backend_buffer_type_t (*get_default_buffer_type)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*synchronize)(ggml_backend_t backend);

        // compute graph with a plan (not used currently)
        // create a new plan for a graph
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph without a plan (async)
        enum ggml_status (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);



        // IMPORTANT: these functions will be removed from the backend interface when the transition to the device interface is complete
        //            new backends should implement the device interface instead

        // These functions are being moved to the device interface
        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_t backend, ggml_backend_buffer_type_t buft);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

#if 0
        // (optional) event synchronization
        // create a new event that can record events on this backend instance
        ggml_backend_event_t (*event_new)         (ggml_backend_t backend);
        void                 (*event_free)        (ggml_backend_event_t event);
        // record an event on the backend instance that created it
        void                 (*event_record)      (ggml_backend_event_t event);
        // wait for an event on on a different backend instance
        void                 (*event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
        // block until an event is recorded
        void                 (*event_synchronize) (ggml_backend_event_t event);
#endif

        // TODO: these functions need to stay in this interface, but the signature may change
        void                 (*event_record)      (ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different backend instance
        void                 (*event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
    };

    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_context_t context;
        ggml_backend_dev_t device;
    };

    struct ggml_backend_event {
        struct ggml_backend_device * device;
        void * context;
    };

    //
    // Backend registry v2
    //

    // TODO: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    // struct ggml_backend_device_props {
    //     const char * name;
    //     const char * description;
    //     size_t total_memory;
    //     size_t available_memory;
    //     // enum ggml_backend_device_type type;
    // };

    // Other properties that may be necessary:
    // - memory type (to determine if buffer_from_host_ptr is desirable)
    // - async support (to determine in llama.cpp if it can be used to improve load times or other details)
    // - event support (to determine if pipeline parallelism can be used)

    struct ggml_backend_device_i {
        // device characteristics
        const char * (*get_name)(ggml_backend_dev_t dev);
        const char * (*get_description)(ggml_backend_dev_t dev);
        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);
        enum ggml_backend_device_type (*get_type)(ggml_backend_dev_t dev);

        // register the backend associated with this device
        ggml_backend_reg_t (*get_backend_reg)(ggml_backend_dev_t dev);

        // backend (stream) initialization
        ggml_backend_t (* init_backend)(ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*buffer_type)(ggml_backend_dev_t dev);

        // host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        // TODO: move to backend reg interface?
        ggml_backend_buffer_type_t (*host_buffer_type)(ggml_backend_dev_t dev);

        // buffer from pointer (optional)
        // create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // (optional) event synchronization
        // create a new event that can record events on this device
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);
    };

    struct ggml_backend_device {
        struct ggml_backend_device_i iface;
        void * context;
    };

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*device_get)(ggml_backend_reg_t reg, size_t index);

        // request the backend to register a device on the device registry based on the params string
        // eg. for RPC, this may be the address of the server
        ggml_backend_dev_t (*add_device)(ggml_backend_reg_t reg, const char * params);

        // backends can add custom functions that are not part of the standard ggml-backend interface
        // backends should define the type of the function in their public header,
        // and applications can use this function to get a pointer to the function
        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);

        // set the log callback for the backend
        void (*set_log_callback)(ggml_backend_reg_t reg, ggml_log_callback log_callback, void * user_data);
    };

    struct ggml_backend_reg {
        // int version; // TODO: for dynamic loading
        struct ggml_backend_reg_i iface;
        void * context;
    };


    // Internal API
    void ggml_backend_register(ggml_backend_reg_t reg);
    void ggml_backend_device_register(ggml_backend_dev_t device);
    // TODO: backends can be loaded as a dynamic library, in which case it needs to export this function
    // typedef ggml_backend_register_t * (*ggml_backend_init)(void);

#ifdef  __cplusplus
}
#endif
