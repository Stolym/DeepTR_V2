$	?,T}f@c=???xs@t^c??ފ?!???!݀@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???!݀@v()?:@1?@ ??~@I7߈?Y?4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails t^c??ފ?H?Sȕz??1?t><K?a?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!c^G????1F?̱??n?Ib?qm???r11*	33333?B@2E
Iterator::Root㥛? ???!??C?eX@)?g??s???1X@?n?WL@:Preprocessing2T
Iterator::Root::ParallelMapV2K?=?U??!u?E]tD@)K?=?U??1u?E]tD@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!@?n?W@@)??H?}M?1@?n?W@@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?.?4!@Q>??j?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?@?Tj!@mS??0%.@!v()?:@	!       "$	?zԼ?d@5A????q@?t><K?a?!?@ ??~@*	!       2	!       :	??T??@dY?CX'@!7߈?Y?4@B	!       J	!       R	!       Z	!       b	!       JGPUb q?.?4!@y>??j?V@