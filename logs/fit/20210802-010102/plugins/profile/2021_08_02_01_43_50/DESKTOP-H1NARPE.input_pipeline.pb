$	?a7??Q@ՈOKX d@?5???v?!z?΅av@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'z?΅av@;?i???<@1??bՠDt@I?ZO@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails fL?gӁ?Oʤ?6 {?1???]/Ma?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?ِf?1?ِf?r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?5???v?1?5???v?r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!S?[????1:?`???t?I?N?j???r11*	43333[v@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?8??m4??!????nJ@)?X?? ??1???LH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?? ???!)?Wf:P;@)?w??#???1r?? ?]/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??_?L??!???|B'@)M?J???18?.??%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?k	??g??!??\??0%@)?V-??1qtdZ?#@:Preprocessing2T
Iterator::Root::ParallelMapV2?HP???!mk??I@)?HP???1mk??I@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??@??ǈ?!???@)??@??ǈ?1???@:Preprocessing2E
Iterator::RoottF??_??!ԩS?N?@)???????1>????	@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipyX?5?;??!???{?P@)HP?sׂ?1??mIn?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceŏ1w-!o?!?}D[???)ŏ1w-!o?1?}D[???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zd?!j???w]??){?G?zd?1j???w]??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?c?!?2?sx??)a2U0*?c?1?2?sx??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!???|B??)??_?LU?1???|B??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?f!?z#@Q	0???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m?t???@?6????)@!;?i???<@	!       "$	n???)7P@B??? b@???]/Ma?!??bՠDt@*	!       2	!       :	e?c]????+?=?@!?ZO@B	!       J	!       R	!       Z	!       b	!       JGPUb q?f!?z#@y	0???V@