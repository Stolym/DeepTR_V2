$	U??? f@G??Bs@?d??7i??!?;??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?;??@vp?71?:@1?ΤM~@I? O!5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?d??7i??Ǻ?????1rQ-"??[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!¾?D???1??Z?a/d?I??%?<??r11*	?????v@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapz?):????!\րfJ@)?<,Ԛ???1?(RLH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map$(~??k??!.Gma?3<@)?Y??ڊ??1]??+c0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??_?L??!?}@?0?'@)?ܵ?|У?1?9% ~?%@:Preprocessing2T
Iterator::Root::ParallelMapV2?W[?????!I?-?z'!@)?W[?????1I?-?z'!@:Preprocessing2E
Iterator::Root'???????!?m]??O(@) ?o_Ή?1#]??Ӡ@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatK?=?U??!C???a@)a??+e??1.?V,@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch/?$???!???[?@)/?$???1???[?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?z?G???!r?1?M@)/n????1 ~????@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!]=??,[??)?????g?1]=??,[??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????g?!]=??,[??)?????g?1]=??,[??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!t??N?r??)Ǻ???f?1t??N?r??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!F}???C??)?~j?t?X?1F}???C??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIPYY??"@Q???KǼV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		?Vд?!@?0????.@!vp?71?:@	!       "$	{?z:?d@???gcTq@rQ-"??[?!?ΤM~@*	!       2	!       :	!??r?.@??;?PZ(@!? O!5@B	!       J	!       R	!       Z	!       b	!       JGPUb qPYY??"@y???KǼV@