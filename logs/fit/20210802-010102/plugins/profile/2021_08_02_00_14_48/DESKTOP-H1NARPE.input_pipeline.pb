$	???Q?e@????r@F?n?1??!?v?6s?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?v?6s?@?yUg?<@1P)?E?}@I?Z??e3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails F?n?1??????Ŋ??1?Nw?x?f?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?Ov3??1?}͑??I#-??#???r11*	    xu@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????o??!?Ӹ??J@)???S???1pl3j?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?=?U???!?w?ti?@)!?rh????1???r0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatp_?Q??!???F^?-@)??+e???1?λxL,@:Preprocessing2E
Iterator::Root????Mb??!?
3??"@)8??d?`??1?8?W,@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!?Ӹ??@)?HP???1??s?i@:Preprocessing2T
Iterator::Root::ParallelMapV2??@??ǈ?!???4.@)??@??ǈ?1???4.@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ǘ????!?4}PX?@)??ǘ????1?4}PX?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipȘ?????!ʾ bעM@)a??+ey?1&7TB???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!5??̕???)?~j?t?h?15??̕???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!TB????)Ǻ???f?1TB????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!??ͨ5???)?J?4a?1??ͨ5???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!?
3????)????MbP?1?
3????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI`}g?9"@QTp?͸V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	b?v?%#@Cy?? {0@!?yUg?<@	!       "$	ͤ`??c@????Cq@?Nw?x?f?!P)?E?}@*	!       2	!       :	W??y??@>卐c&@!?Z??e3@B	!       J	!       R	!       Z	!       b	!       JGPUb q`}g?9"@yTp?͸V@