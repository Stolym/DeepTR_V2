$	L?R?mKf@G?NiNs@??????P?!i:;\??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'i:;\??@v???_w=@1ܜJ I~@I܀?#5@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??????P?1??????P?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!zT?????]?V$&??1?mO???^?r11*	33333u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap;pΈ????!6*???L@)?鷯??1?ʬ?%?J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??V?/???!H0???};@)S?!?uq??1?~???/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatM?J???!?Y?]?0'@)?:pΈ??1???U?x%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat/?$???!??e9?@)?l??????1??F??@:Preprocessing2T
Iterator::Root::ParallelMapV2??0?*??!&????@)??0?*??1&????@:Preprocessing2E
Iterator::Root??JY?8??!GH?_̽@)??ׁsF??1f??|@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?j+??݃?!??'W@)?j+??݃?1??'W@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??ܵ???!i?????P@)???Q?~?1???(?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???_vOn?!2?k?l???)???_vOn?12?k?l???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!V?xyP???)?????g?1V?xyP???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zd?!???????){?G?zd?1???????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!?cŢ????)/n??R?1?cŢ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?ⴐ??"@Q?c?m??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3(????#@?d?Ym1@!v???_w=@	!       "$	i??0d@???.|q@??????P?!ܜJ I~@*	!       2	!       :	Ы?@?6~qrD(@!܀?#5@B	!       J	!       R	!       Z	!       b	!       JGPUb q?ⴐ??"@y?c?m??V@