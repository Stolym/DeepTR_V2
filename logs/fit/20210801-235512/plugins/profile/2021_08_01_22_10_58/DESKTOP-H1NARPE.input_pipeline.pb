$	"????Np@4??_w@Ow?x???!?RE?N?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?RE?N?@????;@1??~j??}@Iq?J[\?3@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!Ow?x???1
?F?c?I[?a/???r11*effff?8@)      ?<2T
Iterator::Root::ParallelMapV2 ?o_Ή?!Y1P?MI@) ?o_Ή?1Y1P?MI@:Preprocessing2E
Iterator::Root?~j?t???!?k??X@)?+e?X??1??ί=?F@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??H?}M?!???h?@)??H?}M?1???h?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(??{3"@Q?F????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????+@?M$?O?3@!????;@	!       "$	G?`ƥm@?*\???t@
?F?c?!??~j??}@*	!       2	!       :$	??V`??#@{?xM,@[?a/???!q?J[\?3@B	!       J	!       R	!       Z	!       b	!       JGPUb q(??{3"@y?F????V@