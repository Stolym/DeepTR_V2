$	gkjFh@5???t@??1ZGU??!??rgF?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'??rgF?@/kb???=@1??5"8d?@I)?A&?5@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??1ZGU????+,??1rQ-"??[?r3"k
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails*.s?,&6???D??2???1$D??b?IǺ?????r11*	fffff&~@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?|a2U??!D??ysJ@)}гY????1?-?jL?E@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map?D?????!Pf]?^?>@)!?rh????1????l7@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch?0?*??!?u?z?!@)?0?*??1?u?z?!@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?-?????!?rv?@)??(????1Zq?$K@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??(????!Zq?$K@)?!??u???1?N??b@:Preprocessing2T
Iterator::Root::ParallelMapV2U???N@??!Ji?m-@)U???N@??1Ji?m-@:Preprocessing2E
Iterator::Root'?Wʢ?!??%?an@)??d?`T??1T(?.V?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipΈ?????!??,u??N@)?j+??݃?1??m? @:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?O??nr?!??????);?O??nr?1??????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??H?}m?!HT?n???)??H?}m?1HT?n???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangea2U0*?c?!aph>???)a2U0*?c?1aph>???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!?b??):??)-C??6Z?1?b??):??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIX"??a"@Q????ȳV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	???}?$@F??? 1@??+,??!/kb???=@	!       "$	J
, ?e@Ճ??[?r@rQ-"??[?!??5"8d?@*	!       2	!       :	p?U̫@?]	Je/)@!)?A&?5@B	!       J	!       R	!       Z	!       b	!       JGPUb qX"??a"@y????ȳV@