$		C??6?e@????r@?9z?ަ??!p?DI?I?@	!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'p?DI?I?@$???8@1cFx{?~@IxG?j??.@r0"a
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails ?9z?ަ????S9?)??1vk???i?r3"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!b?????1ŭ???g?Iy7R???r11*     w@)      0=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap$???????!?,_?JI@)}??b???1????U)G@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map????Q??! f4??:@)??e??a??1?=?m2@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?p=
ף??!?\????!@)u????1??[: @:Preprocessing2?
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatX9??v???!?P?0? @)??j+????1"???^@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?e??a???!ŧ??<P@)??ZӼ???1?JI??$@:Preprocessing2T
Iterator::Root::ParallelMapV2M?O???!?'->?@)M?O???1?'->?@:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch??_vO??!"^?Gr@)??_vO??1"^?Gr@:Preprocessing2E
Iterator::Root?sF????!??ݡ?q @)M?O???1?'->?@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::TensorSliceF%u?k?!??1????)F%u?k?1??1????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Le?!???????)??_?Le?1???????:Preprocessing2?
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?J?4a?!T??p<??)?J?4a?1T??p<??:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::TensorSlice-C??6Z?!?I?????)-C??6Z?1?I?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?H???@Q?~{a?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	`4>?Ȥ @??<Qr?,@!$???8@	!       "$	???A?	d@WL?DnZq@ŭ???g?!cFx{?~@*	!       2	!       :	?9ov~@?L?g!@!xG?j??.@B	!       J	!       R	!       Z	!       b	!       JGPUb q?H???@y?~{a?W@