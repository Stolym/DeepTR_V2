$	]=?f@??k???s@??!????!??|??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??|??@??,??:@1K??~@I?a3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails 3?<Fy???#??^??1?'eRC[?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??!?????}?????1??I???b?r11*	gffff>x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapF%u???!?h?89K@)?(??0??1?D?{^I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?&?W??!'{ZS?:@)??ܵ???1^??x??,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat??0?*??!??-V(@)??_vO??1Ȳ?F&@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatK?=?U??!?g?N??@)y?&1???1P????@:Preprocessing2T
Iterator::Root::ParallelMapV2ˡE?????!??s<?#@)ˡE?????1??s<?#@:Preprocessing2E
Iterator::Root?|a2U??!?9ڑr @)?+e?X??1F ??ނ@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?W?2??!?????OP@)M??St$??1[???N@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetchn????!?Q]?6@)n????1?Q]?6@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range????Mbp?!-	????)????Mbp?1-	????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea??+ei?!l?Q?ג??)a??+ei?1l?Q?ג??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?! ??S?r??)??_?Le?1 ??S?r??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice?~j?t?X?!ÍM????)?~j?t?X?1ÍM????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIP??$!@Q6?}d?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	=G仔
"@?0???!/@?}?????!??,??:@	!       "$	\?????d@?????q@?'eRC[?!K??~@*	!       2	!       :	s????@~?}ӏ`&@!?a3@B	!       J	!       R	!       Z	!       b	!       JGPUb qP??$!@y6?}d?V@