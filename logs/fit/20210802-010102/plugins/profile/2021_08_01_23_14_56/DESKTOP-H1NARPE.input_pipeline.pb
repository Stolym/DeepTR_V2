$	? I	f@?ie|?s@! _B???!`Z?'y??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'`Z?'y??@???M<@1{j????}@I9?t?yF6@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??ù???	?L?n??1X?|[?Tg?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!! _B???1$D??b?I?d??r11*	?????Y|@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?????K??!8#ތx?J@)lxz?,C??1k??VH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?%䃞???!k???S1@@)?~j?t???1?RJ?)5@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatȘ?????!k???Sq&@)?a??4???1;r?s y%@:Preprocessing2T
Iterator::Root::ParallelMapV2䃞ͪϕ?!??2w?@)䃞ͪϕ?1??2w?@:Preprocessing2E
Iterator::Root8??d?`??!Nōj?!@)?l??????1??l?]P@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?W[?????!?????
@)?W[?????1?????
@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat%u???!?^B{	?	@)?{??Pk??1??V?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?ZӼ???! ?ՑkM@)? ?	??1<ݚ)??:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?<,Ԛ?}?!Sq?????)?<,Ԛ?}?1Sq?????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!?ECmb
??)/n??b?1?ECmb
??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??H?}]?!Y?eY?e??)??H?}]?1Y?eY?e??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!?z?@?W??)??_?LU?1?z?@?W??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?+3?#@Q??9??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?iY??"@?Ti??60@!???M<@	!       "$	???G?c@??&tCq@$D??b?!{j????}@*	!       2	!       :	????w?@L)??G?)@!9?t?yF6@B	!       J	!       R	!       Z	!       b	!       JGPUb q?+3?#@y??9??V@