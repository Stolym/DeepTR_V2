$	?>???Q@?????c@X??j??!?9??v@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?9??v@?v?k?.:@1??&t@I?Dg?E8@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?P?l????? !ʇ?1???]/Ma?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsX??j??1X??j??r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?d??1?d??r5"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?j{???1?mO???n?IW?"????r11*	????̔~@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap%u???!QF?)?H@)_?Q???1f?7z`<F@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat[B>?٬??!j?v޴K5@)ݵ?|г??1???R??4@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?:pΈ???!vHC?K8B@)?3??7???1 L?I.@:Preprocessing2T
Iterator::Root::ParallelMapV2ŏ1w-!??!?܉y?@)ŏ1w-!??1?܉y?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?]K?=??!%?XJϾ@)
ףp=
??1m?7??d@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???????!??;C?@)???????1??;C?@:Preprocessing2E
Iterator::Root?0?*???!lO??n @)n????16??? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipio???T??!?#?L?K@);?O??n??1'Y8n??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice;?O??nr?!'Y8n??);?O??nr?1'Y8n??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!?E9V???)	?^)?p?1?E9V???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeŏ1w-!o?!?܉y???)ŏ1w-!o?1?܉y???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!?z????)?~j?t?X?1?z????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???0&"@Q	????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	~u?Y??@???hj'@!?v?k?.:@	!       "$	d?ǼP@?[??kb@???]/Ma?!??&t@*	!       2	!       :	????#?????K$? @!?Dg?E8@B	!       J	!       R	!       Z	!       b	!       JGPUb q???0&"@y	????V@