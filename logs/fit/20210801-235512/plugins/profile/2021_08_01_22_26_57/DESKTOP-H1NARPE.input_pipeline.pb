$	æ??$g@h]?U	t@??E?T???!??eO?Z?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??eO?Z?@???#?i>@1?@?"?@I?0`?U?3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??E?T???:<??Ӹ??1{??h?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!r?CQ?O??1??.??y?I6Vb?????r11*	gffff&H@2T
Iterator::Root::ParallelMapV2????????!L??@B?I@)????????1L??@B?I@:Preprocessing2E
Iterator::RootΈ?????!???J1AS@)?HP???1?K??@B9@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipA??ǘ???!?}?:?6@)A??ǘ???1?}?:?6@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??s)"@Qb???ܺV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7bSԆH$@????J?1@!???#?i>@	!       "$	?Fl??
e@o??@9r@{??h?!?@?"?@*	!       2	!       :	 Ĩ~??@b?b?&@!?0`?U?3@B	!       J	!       R	!       Z	!       b	!       JGPUb q??s)"@yb???ܺV@