$	?*zI?d@?r[???q@?'eRCk?!ςPއ?~@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'ςPއ?~@Nc{-?e8@1?aMe?p|@I?\m??R*@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails  ??q????????ʴ?1{??h?r3"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?'eRCk?1?'eRCk?r11*	     hx@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??? ?r??!!?? ?tH@)??ͪ????1?×???F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map1?*????!M)
%?A@)-????ƻ?15V????;@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat? ?	???!??ۥ??@)???QI??1w?'??K@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?!??u???!?????@)??ݓ????1l"3???@:Preprocessing2T
Iterator::Root::ParallelMapV2?]K?=??!??&2C?@)?]K?=??1??&2C?@:Preprocessing2E
Iterator::Root?b?=y??!?&2C?{@)?g??s???1??=T;?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?0?*??!ޑ???M@)??ǘ????1I*??? @:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchlxz?,C|?!?+??}E??)lxz?,C|?1?+??}E??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!_?>????)?????g?1_?>????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???f?!?iq?????)Ǻ???f?1?iq?????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!fro\????)HP?s?b?1fro\????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!?@?6??)/n??b?1?@?6??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIP?87q?@Qu??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8?t?Q @H꯬- ,@!Nc{-?e8@	!       "$	?
?c??b@???hkp@{??h?!?aMe?p|@*	!       2	!       :	R??ة?@j?y?e@!?\m??R*@B	!       J	!       R	!       Z	!       b	!       JGPUb qP?87q?@yu??W@