$	?!Іi6g@*0/??t@#?=???!?bԵ6h?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?bԵ6h?@????E6?@1???X?@I??]??2@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?iQ??????O??1?h㈵?d?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!#?=???B???D??1?'eRC[?r11*	fffff?y@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap o?ŏ??!???1?=H@)???????1?W?&?F@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???9#J??!iS?"L?9@)?s????1Ԙ??1@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat㥛? ???!(uU?!@)?R?!?u??1???޲? @:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat㥛? ???!(uU?!@)???x?&??1?(?D @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?????!??????O@)??ݓ????1H?D)?&@:Preprocessing2T
Iterator::Root::ParallelMapV2??ͪ?Ֆ?!t)??ާ@)??ͪ?Ֆ?1t)??ާ@:Preprocessing2E
Iterator::Root??_vO??!?I???$@)??A?f??1i? ?K@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch???Q?~?!0???"??)???Q?~?10???"??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlicea2U0*?s?!?i]?2???)a2U0*?s?1?i]?2???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!&?4)N??)?~j?t?h?1&?4)N??:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!?i]?2???)a2U0*?c?1?i]?2???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice??_?LU?!2?O??2??)??_?LU?12?O??2??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?ŔJ(?!@QHg????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?[t???$@?:?  2@B???D??!????E6?@	!       "$	]H?R?"e@?\??Mr@?'eRC[?!???X?@*	!       2	!       :		q?!]?@fq?ci%@!??]??2@B	!       J	!       R	!       Z	!       b	!       JGPUb q?ŔJ(?!@yHg????V@