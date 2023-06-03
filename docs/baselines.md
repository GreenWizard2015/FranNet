# Baseline oracles

The baseline for upscaling and colorization is a naive approach used as a reference point for comparison. Upscaling simply applies a basic upscaling method, such as bilinear or bicubic interpolation, to the input image. Colorization is performed either by using the color version of the image as a reference or by simply returning the grayscale image as the result.

Using the color version of the image as a reference for the oracle would provide an unfair advantage, as it would essentially be using the ground truth information to perform the task. Therefore, the oracle is evaluated with and without using the color image as a reference to assess its performance objectively. As can be seen from the table, even when using the color image as a reference, the results are not perfect due to information loss during resizing.

<table class="myTable">
  <tr>
    <th colspan="5">Entire test dataset</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="2">with color reference</th>
    <th colspan="2">without color reference</th>
  </tr>
  <tr>
    <th>Method</th>
    <th>RGB MSE</th>
    <th>Grayscale MSE</th>
    <th>RGB MSE</th>
    <th>Grayscale MSE</th>
  </tr>
  <tr>
    <td>nearest</td>
    <td>0.00302</td>  <td>0.00297</td>
    <td>0.01256</td>  <td>0.00296</td>
  </tr>
  <tr class='used'>
    <td>bilinear</td>
    <td>0.00149</td>  <td>0.00148</td>
    <td>0.01107</td>  <td>0.00148</td>
  </tr>
  <tr>
    <td>bicubic</td>
    <td>0.00141</td>  <td>0.00140</td>
    <td><b>0.01100</b></td>  <td>0.00140</td>
  </tr>
  <tr>
    <td>area</td>
    <td>0.00174</td>  <td>0.00171</td>
    <td>0.01131</td>  <td>0.00171</td>
  </tr>

  <tr>
    <th colspan="5">Used for validation subset (512 images)</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="2">with color reference</th>
    <th colspan="2">without color reference</th>
  </tr>
  <tr>
    <th>Method</th>
    <th>RGB MSE</th>
    <th>Grayscale MSE</th>
    <th>RGB MSE</th>
    <th>Grayscale MSE</th>
  </tr>
  <tr>
    <td>nearest</td>
    <td>0.00299</td>  <td>0.00293</td>
    <td>0.01337</td>  <td>0.00293</td>
  </tr>
  <tr class='used'>
    <td>bilinear</td>
    <td>0.00148</td>  <td>0.00146</td>
    <td>0.01190</td>  <td>0.00146</td>
  </tr>
  <tr>
    <td>bicubic</td>
    <td><b>0.00139</b></td>  <td><b>0.00138</b></td>
    <td>0.01182</td>  <td><b>0.00138</b></td>
  </tr>
  <tr>
    <td>area</td>
    <td>0.00173</td>  <td>0.00170</td>
    <td>0.01213</td>  <td>0.00170</td>
  </tr>
</table>

(data was collected via `scripts/test-oracle.py`)

In brief, the RGB MSE value of `0.01100` can be considered the primary baseline since it represents the ability to restore color from grayscale images. Achieving a value close to `0.003` is considered close to ideal, indicating high accuracy in restoring the color channels.

Monitoring the Grayscale MSE is also important. Naive upscale gives `0.00297` Grayscale MSE, so if this metric is higher, it means that the network completely fails to capture the color information.

Both the RGB MSE and Grayscale MSE metrics provide valuable insights into the performance of the network.
