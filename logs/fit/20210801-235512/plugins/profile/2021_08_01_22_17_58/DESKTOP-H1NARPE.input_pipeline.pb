$	pi??@f@??K?MEs@? 3??OL?!????y??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'????y??@????>@1Mf???2~@I	?`???4@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails? 3??OL?1? 3??OL?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!n???V??N(D?!T??1?'eRC[?r11*	fffff?G@2T
Iterator::Root::ParallelMapV2?~j?t???!???:?I@)?~j?t???1???:?I@:Preprocessing2E
Iterator::Root???????!?t?c?DX@)Ǻ?????1?gQ?SnG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipǺ???V?!?gQ?Sn@)Ǻ???V?1?gQ?Sn@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIx1???#@Q??&???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|?S?+$@m׹W1@!????>@	!       "$	?|[??!d@???9Poq@? 3??OL?!Mf???2~@*	!       2	!       :	a?٣?@q?u(@!	?`???4@B	!       J	!       R	!       Z	!       b	!       JGPUb qx1???#@y??&???V@