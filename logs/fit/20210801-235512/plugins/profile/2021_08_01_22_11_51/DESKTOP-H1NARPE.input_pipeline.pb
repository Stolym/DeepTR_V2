$	??- ?r@%?a$?pz@ۢ??d??!M??????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'M??????@C?ʠ?DC@1jL??$??@I?o???c8@r0"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!ۢ??d??1? 3??Ol?I?r?蜟??r11*	fffff?@@2E
Iterator::RootaTR'????!?~???X@)?St$????1??????H@:Preprocessing2T
Iterator::Root::ParallelMapV22U0*???!-Y??ՒG@)2U0*???1-Y??ՒG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip????MbP?!b?=p&@)????MbP?1b?=p&@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?ɫ?%@Q͆*g^V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C?ʠ?D3@?	??@;@!C?ʠ?DC@	!       "$	7?x?+?p@?^?=?w@? 3??Ol?!jL??$??@*	!       2	!       :$	??(??p(@^Ωim51@?r?蜟??!?o???c8@B	!       J	!       R	!       Z	!       b	!       JGPUb q?ɫ?%@y͆*g^V@