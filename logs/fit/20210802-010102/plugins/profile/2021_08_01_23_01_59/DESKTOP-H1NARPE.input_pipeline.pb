$	M???f-g@_*?u?t@ҌE??ɨ?!?y9?a?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?y9?a?@]m?????@1???ﮊ@I??K?;?3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??Xm?_??yt#,*???1iUMu_?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!ҌE??ɨ?1?t><K?a?I???հ??r11*	effff??@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??H?}??!??
? I@)W[??????1%?????G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map}??b???!?B?c?@)?y?):???17??k?3@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatT㥛? ??!?3歗+'@)\ A?c̭?1?`Ba?g%@:Preprocessing2T
Iterator::Root::ParallelMapV2;?O??n??!?_?}?z@);?O??n??1?_?}?z@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?R?!?u??!歗+/@)???_vO??1???[??@:Preprocessing2E
Iterator::Root-!?lV??!????%@)䃞ͪϕ?1??[??U@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipK?46??!3??1R	M@)?J?4??1???0???:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch????Mb??!U??????)????Mb??1U??????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?s?!?2=??>??)a2U0*?s?1?2=??>??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!?_?}?z??);?O??nr?1?_?}?z??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice"??u??q?!L(,??M??)"??u??q?1L(,??M??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!١?w???)??_?LU?1١?w???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIhȤW?"@Q?fu~?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??K%@ ???12@!]m?????@	!       "$	?]c*e@?Tm??5r@iUMu_?!???ﮊ@*	!       2	!       :	?y?o?@6Z????&@!??K?;?3@B	!       J	!       R	!       Z	!       b	!       JGPUb qhȤW?"@y?fu~?V@