$	?????e@?w?n?s@??p?????!???'?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'???'?@?#??7@1%Z?xZ?~@I?.6??4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?4}v???C?8
??1.8??_?f?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??p??????N^???1rQ-"??[?r11*	??????v@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??k	????!X??s?I@)???S???1?ut8??G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapݵ?|г??!?k<??;@)?p=
ף??1??%??2@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatr??????!8????#@)m???{???1<	?]"@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?5?;Nѡ?!?*?xN#@)*??Dؠ?1%?e?@"@:Preprocessing2T
Iterator::Root::ParallelMapV2A??ǘ???!?ݐ??@)A??ǘ???1?ݐ??@:Preprocessing2E
Iterator::Root??镲??!Q+?lny"@)A??ǘ???1?ݐ??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?5?;Nс?!?*?xN@)?5?;Nс?1?*?xN@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipd?]K???!`????tO@)???_vO~?1???`?k @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?~j?t?h?!???2???)?~j?t?h?1???2???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!?˄j??)HP?s?b?1?˄j??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceŏ1w-!_?!=????)ŏ1w-!_?1=????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!=????)ŏ1w-!_?1=????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??=??? @Q?N?,??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?܊??@[?;?˃*@?N^???!?#??7@	!       "$	?֭?H*d@?E?B?vq@rQ-"??[?!%Z?xZ?~@*	!       2	!       :	>H<??@??`%\&(@!?.6??4@B	!       J	!       R	!       Z	!       b	!       JGPUb q??=??? @y?N?,??V@