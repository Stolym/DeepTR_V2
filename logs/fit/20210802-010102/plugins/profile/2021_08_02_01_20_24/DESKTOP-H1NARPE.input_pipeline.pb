$	ShiPe@LQ???dr@?5[y????!?up?w?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?up?w?@?&p??9@1.py?l}@I????}+@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?5[y????A?شR??1?mO???^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???Xm>??1??Z
H?o?I??ôo.??r11*	?????Qw@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapP?s???!?Ί#^LK@)??V?/???1>y#???H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?E???Ը?!8&????9@)S?!?uq??1T[Mt?,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??JY?8??!?b??C'@)f??a?֤?1wMm}4?%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatS?!?uq??!T[Mt?@)??@??ǘ?1??<???@:Preprocessing2T
Iterator::Root::ParallelMapV2?b?=y??!v?"=??@)?b?=y??1v?"=??@:Preprocessing2E
Iterator::Root333333??!????$@)_?Qڋ?1{6B?<)@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???<,Ԋ?!????@)???<,Ԋ?1????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipC?i?q???!??????O@)??ǘ????1???5^@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicey?&1?l?!?쇑???)y?&1?l?1?쇑???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!R:Z?F(??)??_vOf?1R:Z?F(??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!???L??)??_?Le?1???L??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice_?Q?[?!{6B?<)??)_?Q?[?1{6B?<)??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI`????@Q???_?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o?UfJC!@?ǧ,?-@!?&p??9@	!       "$	Jf?Cv?c@$??$??p@?mO???^?!.py?l}@*	!       2	!       :	4?ȸ??@W??lx?@!????}+@B	!       J	!       R	!       Z	!       b	!       JGPUb q`????@y???_?W@