$	??cs?f@????s@??IӠhN?!R???;?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'R???;?@??dV??7@1N^???@I?"2@r0"X
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??IӠhN?1??IӠhN?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!?=Զa??1?'eRCk?Iv?ꭁ???r11*	??????x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??ׁsF??!?????L@)Q?|a??1ByN??6J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map4??@????!K????7@)???JY???1]????^(@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?z6?>??!:?@?3'@)?J?4??1??????%@:Preprocessing2T
Iterator::Root::ParallelMapV2u????!^<???l@)u????1^<???l@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?{??Pk??!?*??@@)??&???1?'gpf?@:Preprocessing2E
Iterator::RootǺ?????!?d ?&@)2??%䃎?1{????R@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetcha2U0*???!+1?y?@)a2U0*???1+1?y?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipR???Q??!<P???7P@)?St$????1?׷;1? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice????Mbp?!?N?m?G??)????Mbp?1?N?m?G??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_vOf?!nP?????)??_vOf?1nP?????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!?s?????)HP?s?b?1?s?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!+1?y???)a2U0*?S?1+1?y???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI0?쀄?@Q4???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/????@?1?`:?+@!??dV??7@	!       "$	?????9e@?%F%?ar@??IӠhN?!N^???@*	!       2	!       :	LM??9@<+???$@!?"2@B	!       J	!       R	!       Z	!       b	!       JGPUb q0?쀄?@y4???W@