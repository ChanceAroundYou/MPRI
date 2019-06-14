$base_dir='.\data\accubrainresult'
$image_name='T1_rigid.nii.gz'
$label_name='result_seg_rigid.nii.gz'
$seg_name='MRPI_seg_rigid.nii.gz'
$data_name='data.txt'
Get-ChildItem $base_dir -Recurse -Filter $image_name | ForEach-Object {
    $dir_name=$_.DirectoryName
    Write-Output $dir_name
    python .\main.py -i $dir_name\$image_name -l $dir_name\$label_name -o $dir_name both
}