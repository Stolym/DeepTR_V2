$	?(??.?P@??2?b@vk???y?!,-#??t@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails',-#??t@ ?M??>@1?s]??r@I??????@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ;s	???????߆?1????}rd?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsvk???y?1vk???y?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsvk???y?1vk???y?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?pZ?????1????}rt?IՒ?r0???r11*	33333sE@2T
Iterator::Root::ParallelMapV2?I+???!A?A?I@)?I+???1A?A?I@:Preprocessing2E
Iterator::Root<?R?!???!?I??X@)?0?*???1?y??3mG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!L??z:???)-C??6J?1L??z:???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIp?i?B%@Q2????WV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??3??@?}M\?*@! ?M??>@	!       "$	N1??,N@<|?.I?`@????}rd?!?s]??r@*	!       2	!       :	e????? ??8? ??!??????@B	!       J	!       R	!       Z	!       b	!       JGPUb qp?i?B%@y2????WV@