$	?Z?L)-??#??;8i?R???Tz?!$Di???	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails $Di????~j?t???1?@fg?;e?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??ฌ?z?1??ฌ?z?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?]?pXz?1?]?pXz?r5"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?]?pXz?1?]?pXz?r6"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??!??z?1??!??z?r8"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	?]?pXz?1?]?pXz?r9"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails
R???Tz?1R???Tz?r10*	?????A@2T
Iterator::Root::ParallelMapV2
ףp=
??!)????rP@)
ףp=
??1)????rP@:Preprocessing2E
Iterator::RootL7?A`???!??GpX@)/?$???1;??,??>@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipa2U0*?S?!?}?@)a2U0*?S?1?}?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 22.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI\?9	ą6@Q镱??^S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??E<\??K????r?!?~j?t???	!       "$	'[a;?'x?j?ȶ?W??@fg?;e?!??!??z?*	!       2	!       :	!       B	!       J	!       R	!       Z	!       b	!       JGPUb q\?9	ą6@y镱??^S@