$	??9]?+g@????Zt@-C??6??!H??_`?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'H??_`?@?e??<@1z?@I????KY3@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails -C??6???30??&??1!>???@`?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???2??????? ???1? 3??O\?r11*	?????)t@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapC?i?q???!\1m?:KK@)O@a????1?{?0H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map-C??6??!??????@)??B?iޱ?1G@????5@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat???{????!v??(?5$@)?sF????1??Mn??"@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch"??u????!???S@)"??u????1???S@:Preprocessing2E
Iterator::Root??&???!??֎l?@)-C??6??1?????@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat???QI??!W8???@)?HP???1p?o@A@:Preprocessing2T
Iterator::Root::ParallelMapV2?0?*??!7H?%ߊ	@)?0?*??17H?%ߊ	@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip&S??:??!??ru?N@)S?!?uq{?1ʣ??p? @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice?????g?!^?Έ*???)?????g?1^?Έ*???:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangeHP?s?b?!?K?????)HP?s?b?1?K?????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4a?!???
????)?J?4a?1???
????:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice/n??R?!
Y?ڛ???)/n??R?1
Y?ڛ???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?g?g?L!@Q??k?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	U???*#@L??m?0@?30??&??!?e??<@	!       "$	j??*e@???Tr@? 3??O\?!z?@*	!       2	!       :	e[??d?@?$C??W&@!????KY3@B	!       J	!       R	!       Z	!       b	!       JGPUb q?g?g?L!@y??k?V@