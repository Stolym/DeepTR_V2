$		V? /ne@y?h??r@?<??S???!?>??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?>??@?UJ??9@1??{?V}@IL?[?߶3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ???????*??ٯ?1T?qs*i?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?<??S???}?;l"3??1rQ-"??[?r11*	fffff?w@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap3ı.n???!%"X?p?J@)e?`TR'??1?[;??H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map^?I+??!????7@)??y?)??1F??#P+@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat????<,??!d?!?$@)?&S???1??ϬC#@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?V-??!?|D???"@)*??Dؠ?1?ٻyi!@:Preprocessing2T
Iterator::Root::ParallelMapV2T㥛? ??!<k۳˫ @)T㥛? ??1<k۳˫ @:Preprocessing2E
Iterator::RootV????_??!???B? '@)?HP???1?
5;??	@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipo?ŏ1??!@ĥ?:P@)vq?-??1r$"X? @:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???Q?~?!6	({"???)???Q?~?16	({"???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!,???Ng??)?~j?t?h?1,???Ng??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!:???????)Ǻ???f?1:???????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!H?63??)??_?Le?1H?63??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!H?63??)??_?LU?1H?63??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIؔ???t!@Qe?g?`?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	a???/? @?K?LZ?,@}?;l"3??!?UJ??9@	!       "$	p>u?R?c@?"^g?p@rQ-"??[?!??{?V}@*	!       2	!       :	e?ω*I@4???&@!L?[?߶3@B	!       J	!       R	!       Z	!       b	!       JGPUb qؔ???t!@ye?g?`?V@