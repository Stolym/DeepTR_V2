$	??|7pg@??p???s@??N??!}??O?A?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'}??O?A?@??[??<@1???Mwk@I?Ӝ?Ȁ4@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??N????ĭ???1$D??b?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!6Y??э?X?|[?T??1vk???i?r11*	gfffft@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapq???h ??!Y?'??K@)??_?L??1??U?I@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??(\?µ?!??;??|:@)?I+???1??jAl+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatˡE?????!\b???)@)@?߾???1???6??'@:Preprocessing2T
Iterator::Root::ParallelMapV2?+e?X??!?-뜉k@)?+e?X??1?-뜉k@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?W[?????!????r?@)???<,Ԋ?1??9>T@:Preprocessing2E
Iterator::Roote?X???!&KBy?%@)??0?*??1?<V??j@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchΈ?????!?9?N/@)Έ?????1?9?N/@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??W?2???!=V??j]O@)9??v??z?1?K3U4 @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!,???????)Ǻ???f?1,???????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range??_vOf?!?JQ????)??_vOf?1?JQ????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!?-]?????)????Mb`?1?-]?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!?i	?????)a2U0*?S?1?i	?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI(?.??!@Q?)z???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?????U#@m"M??0@X?|[?T??!??[??<@	!       "$	?[X7^?d@???X?#r@$D??b?!???Mwk@*	!       2	!       :	,?{?`V@?+?Ƭ'@!?Ӝ?Ȁ4@B	!       J	!       R	!       Z	!       b	!       JGPUb q(?.??!@y?)z???V@