$		???f@,?-?s@!?J͎?!5??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'5??@??G6W?:@1|?ڥ?f@I]??a.@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails !?J͎??'eRC??1? 3??O\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???'???1?t><K?q?I!Y?n??r11*53333?x@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapX?5?;N??!?V?YI@)I??&??1?~ ??HG@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map~8gDi??!??L,??@)EGr????1?]K?ݩ7@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?X?? ??!~?(??@)_?Qڛ?1?bᇳ?@:Preprocessing2T
Iterator::Root::ParallelMapV246<?R??!?:??@)46<?R??1?:??@:Preprocessing2E
Iterator::RootDio??ɤ?!?????$@)U???N@??1?@Tm@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??%䃞??!=??&;PM@);?O??n??1??	8?=@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat2U0*???!?'?V?@)_?Qڋ?1?bᇳ?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?0?*??!a???i?@)?0?*??1a???i?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_?Le?!?<?M??)??_?Le?1?<?M??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?J?4a?!?D???)?J?4a?1?D???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!?D???)?J?4a?1?D???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!???1 7??)????MbP?1???1 7??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI p ???@Q??o?pW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????!@?????#/@!??G6W?:@	!       "$	?md1D?d@??n-!r@? 3??O\?!|?ڥ?f@*	!       2	!       :	??c?@?f??Y!@!]??a.@B	!       J	!       R	!       Z	!       b	!       JGPUb q p ???@y??o?pW@