$	Q???xg@?z^?(t@?Nw?x?f?!5?BX?U?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'5?BX?U?@Z-??D&>@1?fe??@IzPP?V3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?Nw?x?f?fL?g?Q?1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!l????ߡ?1fL?g?a?I?~?T? ?r11*	gffff?A@2T
Iterator::Root::ParallelMapV2?Q?????!䛡??pH@)?Q?????1䛡??pH@:Preprocessing2E
Iterator::Roote?X???!?ir?y)X@)???????1?7Ck??G@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipa2U0*?S?!]Ų???
@)a2U0*?S?1]Ų???
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?{~?!@Q?>P~?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ݸ?$@H?nhh1@!Z-??D&>@	!       "$	P????e@K??4>r@rQ-"??[?!?fe??@*	!       2	!       :	L??Пc@????T?%@!zPP?V3@B	!       J	!       R	!       Z	!       b	!       JGPUb q?{~?!@y?>P~?V@