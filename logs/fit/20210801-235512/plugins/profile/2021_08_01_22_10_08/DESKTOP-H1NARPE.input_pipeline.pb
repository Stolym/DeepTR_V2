$	|G?	??g@p$Y҇t@W?f?"??!?tF?́@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?tF?́@??n-?@@1n?HJ?B?@I:x&4I?/@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails W?f?"???Fˁj??1?y?Cn?k?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????#??|_\????1-C??6j?r11*	333333>@2T
Iterator::Root::ParallelMapV2X9??v???!%?S??I@)X9??v???1%?S??I@:Preprocessing2E
Iterator::Root%u???!?Y??vVX@)??Pk?w??1??dG@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-C??6J?!??Tb*1@)-C??6J?1??Tb*1@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?¾?T?!@Q?'?dU?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??? '@	?^?2@?Fˁj??!??n-?@@	!       "$	???L??e@?*e??r@-C??6j?!n?HJ?B?@*	!       2	!       :	|?x?B@R???i"@!:x&4I?/@B	!       J	!       R	!       Z	!       b	!       JGPUb q?¾?T?!@y?'?dU?V@