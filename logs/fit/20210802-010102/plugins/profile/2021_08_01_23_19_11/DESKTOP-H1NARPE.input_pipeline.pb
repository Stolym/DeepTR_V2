$	V0*??iV@a/?}?Nf@??^?S?!???lUv@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???lUv@???N?H=@1?ʿ?W@t@I?̔??"@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?o?DIH??Úʢ????1rQ-"??[?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??^?S?1??^?S?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!䠄????1?_>Y1\m?I?D?k????r11*	?????t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ׁsF??!lPFp?H@)&S????1?=????F@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??y?)??!?:O???/@)??@??Ǩ?1h)???.@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?? ???!?T??c>@)$????ۧ?1??X?6?,@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatHP?s??!??|m?}@)??ׁsF??1lPFp?@:Preprocessing2E
Iterator::Root?Q?????!/Y`Z??%@)/n????1?c?!??@:Preprocessing2T
Iterator::Root::ParallelMapV2?5?;Nё?!oN???@)?5?;Nё?1oN???@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch	?^)ˀ?!??Bȃg@)	?^)ˀ?1??Bȃg@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??0?*??!?????\M@)???Q?~?1qL?ߚ?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???_vOn?!?6?P?i??)???_vOn?1?6?P?i??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!?iR;????)a??+ei?1?iR;????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!?qL????)??_vOf?1?qL????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?g)?????)Ǻ???V?1?g)?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIh???G#@Q3j???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??h:;M@?U??G-@!???N?H=@	!       "$	?]???@T@?& ?E@d@rQ-"??[?!?ʿ?W@t@*	!       2	!       :	?ݑ?????u???p??!?̔??"@B	!       J	!       R	!       Z	!       b	!       JGPUb qh???G#@y3j???V@