$	???"GV@AY??Df@[???iv?!.???qEv@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'.???qEv@??a?1?9@1܄{e?Et@Iͱ???@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails G!ɬ????l@??r??1rQ-"??[?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails[???iv?1[???iv?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!4??`??/PR`L??1K?8???\?r11*	    ?u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap^K?=???!?ځ?vH@)?1??%???1?B???F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapV-?????!R?g??;@)q=
ףp??1??????0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat;?O??n??!Nozӛ?$@)??镲??1??7??M#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??ܵ?|??!??????"@)??H?}??1Y?B? @:Preprocessing2T
Iterator::Root::ParallelMapV2M??St$??!???(?3@)M??St$??1???(?3@:Preprocessing2E
Iterator::Root??_vO??!??,d!)@)?0?*??1~F??Q?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch	?^)ˀ?!0?̵@)	?^)ˀ?10?̵@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???????!r??\;0N@)?<,Ԛ?}?1̵s? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}m?!Y?B???)??H?}m?1Y?B???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!?%~F???)_?Q?k?1?%~F???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!??,d!??)??_vOf?1??,d!??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!ہ?v`???)-C??6Z?1ہ?v`???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???? "@Q? ?o??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??=@??@???)??)@!??a?1?9@	!       "$	(E+??ET@U??4?Ed@rQ-"??[?!܄{e?Et@*	!       2	!       :	ͱ?????ͱ???@!ͱ???@B	!       J	!       R	!       Z	!       b	!       JGPUb q???? "@y? ?o??V@