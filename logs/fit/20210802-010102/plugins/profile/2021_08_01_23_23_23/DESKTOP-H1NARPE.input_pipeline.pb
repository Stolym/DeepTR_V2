$	??&?lf@{3?7ks@??IӠhN?!}???6р@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'}???6р@ҫJC?<@1???Zt?~@I?i?WV>4@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??IӠhN?1??IӠhN?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???4????%?<??1? 3??O\?r11*	??????w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapԚ?????!?5???vI@)?Zd;??1?k}p?G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?t?V??!?lH?@)???~?:??1\V???0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatK?46??!??2?,@)?'????1?Ph79*@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?
F%u??!???@)w-!?l??15??
?@:Preprocessing2T
Iterator::Root::ParallelMapV2r??????!iMex?@)r??????1iMex?@:Preprocessing2E
Iterator::Root????Mb??!6??? @)???QI??1r??%??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?u?????!n??JcNN@)???????1 (?nI@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch????Mb??!6??? @)????Mb??16??? @:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range{?G?zt?!??${????){?G?zt?1??${????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice???_vOn?!?p?E????)???_vOn?1?p?E????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!?3y?O??)y?&1?l?1?3y?O??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!?~?-	s??)Ǻ???V?1?~?-	s??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?f'?e>"@Q/?E3?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?_?`?;#@??-O??0@!ҫJC?<@	!       "$	?G}?`d@נ??q@??IӠhN?!???Zt?~@*	!       2	!       :	u7Ou??@y??2`'@!?i?WV>4@B	!       J	!       R	!       Z	!       b	!       JGPUb q?f'?e>"@y/?E3?V@