$	ł?k?f@gY????s@=?E~???!|??8G?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'|??8G?@v??$?H=@1?ۃ?~@I?J???.6@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails I,)w?????x???1????1ʃ?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!=?E~???}?;l"3??1?mO???^?r11*	     Xu@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ܵ?|??!????L@)-!?lV??1?ˠ?2hH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map}гY????!??H?-?>@)*??Dذ?1?]m?D3@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?N@aã?!j?l?,?&@)?&S???1UWE?UQ%@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??_?L??!??sa?\@)??_?L??1??sa?\@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat????Mb??!,??J??@)?!??u???1????@:Preprocessing2T
Iterator::Root::ParallelMapV2?(??0??!͗A?e?@)?(??0??1͗A?e?@:Preprocessing2E
Iterator::Root?~j?t???!??qp|@)g??j+???1?K???h@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipf?c]?F??!Rf͔Y3O@)?????w?1?i?l?,??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!???h?<??)Ǻ???f?1???h?<??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/n??b?!J?uRl???)/n??b?1J?uRl???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!7G????)ŏ1w-!_?17G????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!,??J????)????MbP?1,??J????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?X???"@Q?-??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	 ??q??#@{r???0@?x???!v??$?H=@	!       "$	???Ԥd@?iEr??q@?mO???^?!?ۃ?~@*	!       2	!       :	??4??@c/?x??)@!?J???.6@B	!       J	!       R	!       Z	!       b	!       JGPUb q?X???"@y?-??V@