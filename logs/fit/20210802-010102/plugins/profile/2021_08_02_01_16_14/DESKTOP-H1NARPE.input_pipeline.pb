$	???W?f@????s@?V횐ֈ?!m?IF???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'm?IF???@??6???8@1?~??@J@I?D?<92@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?V횐ֈ????|	??1??IӠh^?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!עh[͊??B,cC??1? 3??O\?r11*	????̼w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapı.n???!?%?~??L@)?g??s???1'&???SF@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map????!??f?6@)??+e???1?6???)@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?????K??!_.?-??'@)?????K??1_.?-??'@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat@?߾???!??Vt#+$@)?V-??1?,_:??"@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!#???En"@)??ܵ?|??1?៚?? @:Preprocessing2E
Iterator::Root??ǘ????!f$V??!@){?G?z??1d?O@:Preprocessing2T
Iterator::Root::ParallelMapV2a??+e??!???pw
@)a??+e??1???pw
@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipy?&1???!?:jĭ%Q@)S?!?uq{?1z?͞?9??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!JEz?c???)Ǻ???f?1JEz?c???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!JEz?c???)Ǻ???f?1JEz?c???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_vOf?!?0?Y????)??_vOf?1?0?Y????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!5??q????)????MbP?15??q????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?S?T??@Q?Z??FW@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?I?? @տ?<??,@???|	??!??6???8@	!       "$	??j?4?d@t?zr?r@? 3??O\?!?~??@J@*	!       2	!       :	?YPL@????
%@!?D?<92@B	!       J	!       R	!       Z	!       b	!       JGPUb q?S?T??@y?Z??FW@