$	?V????f@?1??D?s@??!????!b?G??@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'b?G??@r4GV~;@1??;??a@I???`-1@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ??!????}?;l"3??1? 3??O\?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!??H٢??~??7???1?'eRC[?r11*	fffff&u@2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap"?uq??![oxT?K@)u?V??1?8??8NH@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map???????!??
br?=@)W[??재?1???sׄ0@:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat?I+???!????5*@)ˡE?????1s?HO;(@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!?????F@)X9??v???1vd?NR@:Preprocessing2T
Iterator::Root::ParallelMapV2?ZӼ???!^ =??@)?ZӼ???1^ =??@:Preprocessing2E
Iterator::Root}гY????!???Q??@)tF??_??1?V')?!@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSliceM??St$??!%n?KŶ
@)M??St$??1%n?KŶ
@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??~j?t??!??#?hu@)??~j?t??1??#?hu@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???V?/??!???
bO@)9??v??z?1?U>?????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?~j?t?h?!?'Ni^??)?~j?t?h?1?'Ni^??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSlice??_?Le?!?e????)??_?Le?1?e????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zd?!?˹k???){?G?zd?1?˹k???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???7a @Q??e??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	㯶O?h"@?W/?.@}?;l"3??!r4GV~;@	!       "$	?? Z??d@???*Gr@?'eRC[?!??;??a@*	!       2	!       :	?ǆ[+?@??rS??#@!???`-1@B	!       J	!       R	!       Z	!       b	!       JGPUb q???7a @y??e??V@