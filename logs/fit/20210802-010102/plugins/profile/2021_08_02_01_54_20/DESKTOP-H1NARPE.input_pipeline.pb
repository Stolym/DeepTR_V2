$	;???c?f@kA??H?s@R???Tz?!?z???@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?z???@?G?`?<@1?a???@I????576@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails R???Tz???̔??r?1K?8???\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!????y??$?@???1?mO???^?r11*	?????<{@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap6<?R?!??! ??K?K@)-???????1???ې?H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?uq???!??G4A?:@)?lV}????1opAq?0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat46<?R??!a??Y$@)?1w-!??1?f2?)	"@:Preprocessing2T
Iterator::Root::ParallelMapV2ݵ?|г??!ǡ?R?	@)ݵ?|г??1ǡ?R?	@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?!??u???!??[?(?@)?HP???1-?,?Ee@:Preprocessing2E
Iterator::Root???Mb??!?䡊̑%@)w-!?l??1.(???@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?+e?X??!aB?Q?@)?+e?X??1aB?Q?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip`vOj??!q?3C?7O@)?? ?rh??1?k?[?4??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceǺ???v?!.S?T???)Ǻ???v?1.S?T???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range"??u??q?!+[?M????)"??u??q?1+[?M????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!o?!,w????)ŏ1w-!o?1,w????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceǺ???V?!.S?T???)Ǻ???V?1.S?T???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?????"@Qm_B???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	KV??#@???^?{0@??̔??r?!?G?`?<@	!       "$	۝Ȧo?d@o'?fU?q@K?8???\?!?a???@*	!       2	!       :	?;YW??@?A?)@!????576@B	!       J	!       R	!       Z	!       b	!       JGPUb q?????"@ym_B???V@