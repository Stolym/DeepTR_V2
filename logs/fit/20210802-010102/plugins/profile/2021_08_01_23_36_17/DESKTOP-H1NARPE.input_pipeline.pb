$	?NJ.l?d@?́?q@?ܚt["??!?q6?'@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?q6?'@?҇.?O=@1?M??;0|@I?nJy?(2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?ܚt["??¾?D???16w??\?f?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????2??1??.??i?I???,&??r11*	??????x@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?q??????!???b??G@)?e??a???1=v????F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map"??u????!X??R??A@)?q??۸?1??Gt??8@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat#??~j???!Zcr?$@)vOjM??1alDh?:#@:Preprocessing2E
Iterator::Root??ܵ?|??!?<?D?l @)????<,??1,?[?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipjM??St??!}G???XL@)"??u????1X??R??@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!¬?t?\@)????Mb??1:D{??R@:Preprocessing2T
Iterator::Root::ParallelMapV2????????!???Z??	@)????????1???Z??	@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch9??v??z?!?N?????)9??v??z?1?N?????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeǺ???f?!?_y?????)Ǻ???f?1?_y?????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???f?!?_y?????)Ǻ???f?1?_y?????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mb`?!:D{??R??)????Mb`?1:D{??R??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice????MbP?!:D{??R??)????MbP?1:D{??R??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI0?؜u#@Q?9?dL?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?
j??#@?{?j??0@!?҇.?O=@	!       "$	?ӕ??b@Ҏ6bDFp@6w??\?f?!?M??;0|@*	!       2	!       :	??:h?@%$?,HX$@!?nJy?(2@B	!       J	!       R	!       Z	!       b	!       JGPUb q0?؜u#@y?9?dL?V@