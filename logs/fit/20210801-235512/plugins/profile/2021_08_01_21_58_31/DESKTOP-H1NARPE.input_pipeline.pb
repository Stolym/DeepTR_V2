$	獓?<?i@?KU?QCv@? 3??OL?!]?@?H?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails']?@?H?@|??8G?:@1;M????@I?N???1@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails? 3??OL?1? 3??OL?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!^?c@?z??[??8?	??1? 3??O\?r11*	?????E@2T
Iterator::Root::ParallelMapV2??A?f??!DP??d?H@)??A?f??1DP??d?H@:Preprocessing2E
Iterator::Root0L?
F%??!? ?b?X@)??ZӼ???1?[u`:H@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!D}?w_g??)-C??6J?1D}?w_g??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI@?N!??@Q\?m?2W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3h????!@Ha???.@!|??8G?:@	!       "$	py???g@?L?t@? 3??OL?!;M????@*	!       2	!       :	o?ݳn@I?aK$@!?N???1@B	!       J	!       R	!       Z	!       b	!       JGPUb q@?N!??@y\?m?2W@