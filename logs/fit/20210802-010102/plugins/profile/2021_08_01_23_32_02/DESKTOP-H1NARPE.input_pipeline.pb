$	g??}?f@???(??s@?Z?QfS?!<-?p?5?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'<-?p?5?@T???r?:@1?c??3f@I?n?!?Q5@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?Z?QfS?1?Z?QfS?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????Ό??1?u??t?In???x??r11*	33333Cv@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?Q?|??!???١?K@)M??St$??1o?H.aI@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapE???JY??!7??[:?8@)5?8EGr??1Y????+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatU???N@??!j橚%@)P?s???1
??? ?#@:Preprocessing2E
Iterator::Root???x?&??!?_??E?"@)8??d?`??1????X@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatM?O???! ?
?@)??d?`T??1? σ?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip6<?R?!??!w?<???P@)??ܵ?|??1???7i@:Preprocessing2T
Iterator::Root::ParallelMapV2_?Qڋ?!???bm?@)_?Qڋ?1???bm?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???S㥋?!璺v?Q@)???S㥋?1璺v?Q@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice_?Q?k?!???bm???)_?Q?k?1???bm???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!??ќ???)HP?s?b?1??ќ???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP?s?b?!??ќ???)HP?s?b?1??ќ???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!??CB?'??)Ǻ???V?1??CB?'??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??7?!@Q??<Y?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???}L?!@+???W*/@!T???r?:@	!       "$	%?H???d@H??? r@?Z?QfS?!?c??3f@*	!       2	!       :	??b?@??????'@!?n?!?Q5@B	!       J	!       R	!       Z	!       b	!       JGPUb q??7?!@y??<Y?V@