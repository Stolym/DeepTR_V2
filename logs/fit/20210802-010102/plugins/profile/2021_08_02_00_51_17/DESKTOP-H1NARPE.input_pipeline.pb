$	?*?^Qe@?<?DPur@i5$?????!p????@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'p????@W\???8@1%?z?p}@I??x"??/@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails i5$????????.5B??1??I???b?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!n?HJz??1?'eRCk?I???"Ʀ?r11*	??????t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??	h"l??!e{?ݘL@)+????1??????J@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??k	????!??;???;@)?~j?t???1*ۻ??,@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?lV}???!>?o
?*@)#??~j???1??n??G(@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?l??????!?#6?.@)? ?	???1]h?+w@:Preprocessing2T
Iterator::Root::ParallelMapV2_?Qڋ?!!?ᐔN@)_?Qڋ?1!?ᐔN@:Preprocessing2E
Iterator::Roota??+e??!??}Sx?@)Ǻ?????1??7???
@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch	?^)ˀ?!?戁?@)	?^)ˀ?1?戁?@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??ʡE???!;??*z9P@)??H?}}?1????$D@:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeŏ1w-!o?!??e?9??)ŏ1w-!o?1??e?9??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora??+ei?!??}Sx???)a??+ei?1??}Sx???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!d???W???)?????g?1d???W???:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlicea2U0*?S?!??/r???)a2U0*?S?1??/r???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??,??@Q?0}??W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	dw??? @??<(??,@!W\???8@	!       "$	?? ?!?c@Y?????p@??I???b?!%?z?p}@*	!       2	!       :	+?gz?!@J?Su9"@!??x"??/@B	!       J	!       R	!       Z	!       b	!       JGPUb q??,??@y?0}??W@