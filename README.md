# A BATCH_SIZE wise conpersion for VGG and Wide-RasNet
> This is joint developed by Liuziying Chen, Zhenghan Fang, Bingbing Cao and Junfeng Hu

For the final project of MATHEMATICS AT PEKING UNIVERSITY a PKU-PSU joint proglam, we did a BATCH_SIZE wise conpersion for VGG and Wide-RasNet. To findout which BATCH_SIZE will be best for train model. We used BATCH_SIZE of 32 64 128 and 256.

VGG

![](/img/vgg_train_loss.png)
![](/img/vgg_test_loss.png)
![](/img/vgg_train_acc.png)
![](/img/vgg_test_acc.png)

Ras Net

![](/img/ras_train_loss.png)
![](/img/ras_test_loss.png)
![](/img/ras_train_acc.png)
![](/img/ras_test_acc.png)

It seams that 256 out perform other batch size. One thing interesting is that we find lower learning rate helps acc and loss. We lower the learning rate at 75 and 100. The change on 75 is most outstanding, while the change on 100 isn't much.

## Installation

OS X & Linux & Windows:

Virtual Environments is recommended

```sh
cd PATH_TO_PROJECT
pip install virtualenv
virtualenv MY_PROJECT
pip install -r requirements.txt
```

## Usage example

You can easily replace the model and batch size to see how your model perform.


## Meta

Liuziying Chen   - [GitHub Link](https://github.com/nixiechennixiechen/)

Zhenghan Fang     – [GitHub Link](https://github.com/ReLRail/)

Bingbing Cao

Junfeng Hu

Distributed under the MIT license. See ``LICENSE`` for more information.

## Contributing

1. Fork it (<https://github.com/ReLRail/MATH497/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
