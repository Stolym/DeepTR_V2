$	????(@?????<?o??8?*5{?u?!???O????	!       "a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???O????? 3??O??1vk???i?r3"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?`???1?`???r4"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?ِf?1?ِf?r5"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8?*5{?u?18?*5{?u?r6*	effffy@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?3??7???!=?CGJ@)9??m4???1?K???F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?	h"lx??!?hk??9@)d?]K???1f!m?NI,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatEGr????!??b;.L'@)w-!?l??1???\??%@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!rX? /?@)lxz?,C??1?k???@:Preprocessing2T
Iterator::Root::ParallelMapV2J+???!82p?q@)J+???182p?q@:Preprocessing2E
Iterator::Root???JY???!?V??o?'@)g??j+???1?????X@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?ZӼ???!(?V@)?ZӼ???1(?V@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::PrefetchǺ?????!|??̆Y@)Ǻ?????1|??̆Y@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?q??????!w'??D!O@)vq?-??1??ssp???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_?Q?k?!??f?#??)_?Q?k?1??f?#??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?????g?!?Ε??%??)?????g?1?Ε??%??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice_?Q?[?!??f?#??)_?Q?[?1??f?#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 69.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??R?yQ@Q????>@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	? 3??O??? 3??O??!? 3??O??	!       "$	1]??ax???u?m?a?vk???i?!?`???*	!       2	!       :	!       B	!       J	!       R	!       Z	!       b	!       JGPUb q??R?yQ@y????>@