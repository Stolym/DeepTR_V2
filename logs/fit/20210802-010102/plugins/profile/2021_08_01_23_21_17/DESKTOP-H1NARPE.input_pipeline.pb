$	?g??<Ee@,`??Wkr@b?o???!?IEcm?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?IEcm?@$d ?.s9@1ɰ?7?}@I?*?샤3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails b?o???[???i??1$D??b?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!_?Qڋ?ŭ??ڇ?1??Z
H?_?r11*	33333?t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapTt$?????!?|6?A?J@)?>W[????1??zJoH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map鷯猸?!?>c?<@)??y?)??1?k"???.@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatǺ?????!r?*+@)??g??s??1+?'??C)@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatDio??ɔ?!?h?z@)?Q?????1??k?x@:Preprocessing2E
Iterator::Root????o??!?:.e??"@)?o_???1??l??#@:Preprocessing2T
Iterator::Root::ParallelMapV2y?&1???!???I??@)y?&1???1???I??@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??ׁsF??!?????@)??ׁsF??1?????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?m4??@??!?Q2$??N@)?? ?rh??1?Q???@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceF%u?k?!?o?????)F%u?k?1?o?????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!Z$?k"???)?????g?1Z$?k"???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???f?!r?*??)Ǻ???f?1r?*??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!,z?/gK??)????MbP?1,z?/gK??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(??Dڬ!@Q?`c?d?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??S?O? @5{s??_-@[???i??!$d ?.s9@	!       "$	?ʾ+dc@?m?Y??p@??Z
H?_?!ɰ?7?}@*	!       2	!       :	?8??0@??N?m?&@!?*?샤3@B	!       J	!       R	!       Z	!       b	!       JGPUb q(??Dڬ!@y?`c?d?V@