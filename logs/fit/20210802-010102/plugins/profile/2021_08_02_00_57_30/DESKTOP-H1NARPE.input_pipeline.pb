$	" ??w?Q@???T5?c@??IӠh~?!}???K9v@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'}???K9v@H?)s??<@1.=??I)t@I??J??@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails *6?u?!??????y7??1a2U0*?c?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??Z
H??1??Z
H??r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??IӠh~?1??IӠh~?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!y<-?p???1hY????p?I E??????r11*	?????w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapP?s???!?x? ٝK@)?O??e??1???3/?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??|?5^??!8쳬?;@)mV}??b??1?29??0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??|гY??!`?em'?&@)?ܵ?|У?1??p	??$@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatS?!?uq??!?;??6@)g??j+???1??R.a@:Preprocessing2T
Iterator::Root::ParallelMapV2???QI??!??q?@)???QI??1??q?@:Preprocessing2E
Iterator::Root??q????!)? ?,@) ?o_Ή?1jE??T@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?ׁsF???!??"?1P@)?J?4??1??_?8@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?J?4??!??_?8@)?J?4??1??_?8@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!s??G???)_?Q?k?1s??G???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice-C??6j?!??????)-C??6j?1??????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!?LN????)?~j?t?h?1?LN????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!?LN????)?~j?t?X?1?LN????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?)?#?/#@Q?:???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?(")??)@!H?)s??<@	!       "$	?5~<O!P@«??Qb@a2U0*?c?!.=??I)t@*	!       2	!       :	ͬJ?A.??lv?]????!??J??@B	!       J	!       R	!       Z	!       b	!       JGPUb q?)?#?/#@y?:???V@