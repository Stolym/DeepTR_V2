$	??4y?f@f?P?f?s@]?E?~U?!???;???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???;???@??V`??;@1/1??W?~@I???5@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails]?E?~U?1]?E?~U?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!? ?w?~??1:?`???d?Iw? ݗ3??r11*	fffff?D@2T
Iterator::Root::ParallelMapV2Ǻ????!n??0X?R@)Ǻ????1n??0X?R@:Preprocessing2E
Iterator::RootˡE?????!?f???X@)?j+??݃?1ց?{?47@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!G?<???)-C??6J?1G?<???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?N???"@Q0v	%¼V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??9@0?"@&f??? 0@!??V`??;@	!       "$	;?????d@p? )d?q@]?E?~U?!/1??W?~@*	!       2	!       :	h???@բ?U?(@!???5@B	!       J	!       R	!       Z	!       b	!       JGPUb q?N???"@y0v	%¼V@