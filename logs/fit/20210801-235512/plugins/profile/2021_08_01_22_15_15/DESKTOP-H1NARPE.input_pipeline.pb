$	IQg??(X@2??8N&h@?'eRC{?!?x#?&x@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?x#?&x@???,<@1Nd??v@If1????@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ٲ|]?????ʠ??D??1X?|[?Tg?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?'eRC{?1?'eRC{?r4"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!)??q??1?y?Cn?k?I?k?????r11*	    ?>@2T
Iterator::Root::ParallelMapV2?0?*???!.?u?yP@)?0?*???1.?u?yP@:Preprocessing2E
Iterator::Root???B?i??!z???!XX@)a2U0*???1/?u?y?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!?????@)-C??6J?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIHfy?S!@Q7<Ӱ??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b???X<@?#???&,@!???,<@	!       "$		ȳV@?|?r?f@X?|[?Tg?!Nd??v@*	!       2	!       :	=??+?????Ҿ?@!f1????@B	!       J	!       R	!       Z	!       b	!       JGPUb qHfy?S!@y7<Ӱ??V@