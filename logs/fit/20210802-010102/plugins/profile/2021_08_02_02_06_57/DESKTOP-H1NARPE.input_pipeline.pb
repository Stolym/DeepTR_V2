$	/??w?Oe@?=etr@??.??Y?!??PM	?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??PM	?@?y7?9@1?3/?]p}@IBȗP?,@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??.??Y?1??.??Y?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???$?????z????1rQ-"??[?r11*	??????w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapjM????!Ғ?p&L@)3ı.n???1φ???J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??@??Ǹ?!?gE<9@)??HP??1?J??,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatsh??|???!??rm?%@)?ܵ?|У?1_?Poa-$@:Preprocessing2T
Iterator::Root::ParallelMapV22U0*???!b1yF_ @)2U0*???1b1yF_ @:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat????<,??!Kٮϊ@)????Mb??1?f???@:Preprocessing2E
Iterator::Root?J?4??!D????x&@)g??j+???1?/2??f@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??_?L??!v??Qs?@)??_?L??1v??Qs?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipo?ŏ1??!??{???O@)?5?;Nс?1	???$@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???_vOn?!
$?j????)???_vOn?1
$?j????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!pa-*1??)?????g?1pa-*1??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!)??[??)Ǻ???f?1)??[??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice_?Q?[?!{???\??)_?Q?[?1{???\??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIP?6???@Q????W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f?K!@?????-@!?y7?9@	!       "$	=?CG?c@?O??p@??.??Y?!?3/?]p}@*	!       2	!       :	?0e?P@/7?c? @!BȗP?,@B	!       J	!       R	!       Z	!       b	!       JGPUb qP?6???@y????W@