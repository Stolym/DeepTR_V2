$	?c??f@M?
C&?s@?6?xͫ??!{????"?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'{????"?@???3*9@13ı.?W@I4?i???5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?6?xͫ??????z??1??????`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?=%????1
?F?c?I?7? ???r11*	gffffFz@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??Pk?w??!c?s?tsJ@)z6?>W[??1ƩEƖ?G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map[????<??!?g??<:@)~??k	???1dY?z??/@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?I+???!??
???$@)??D????1??#*)u#@:Preprocessing2T
Iterator::Root::ParallelMapV2??0?*??!?K?nt@)??0?*??1?K?nt@:Preprocessing2E
Iterator::RootEGr????!?,X"?7&@)??&???1??+??@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatL7?A`???!|f@)+??????1?	Í@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?&S???!Zw?Q@)?&S???1Zw?Q@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???S㥋?!8'???	@)???S㥋?18'???	@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?q?????!?p?i?SO@)????Mb??1Ɩ@{r??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice;?O??nr?!??Te ??);?O??nr?1??Te ??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea??+ei?!@{r????)a??+ei?1@{r????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!lE?̕[??)-C??6Z?1lE?̕[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?_q1d!@Q??y??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??ˬ? @???1-@!???3*9@	!       "$	??+?d@Ѕtr@??????`?!3ı.?W@*	!       2	!       :	o??0r?@??ӌ	)@!4?i???5@B	!       J	!       R	!       Z	!       b	!       JGPUb q?_q1d!@y??y??V@