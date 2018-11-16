#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QMessageBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , ticksCounter(0)
    , odomImpPerSec(0)
    , totalDst_m(0)
{
    ui->setupUi(this);

    // Leave only 'close' button on window.
    setWindowFlags(Qt::WindowCloseButtonHint);

    // Fill up cbx with COM ports names.
    ComPortUtils *portUtils = new ComPortUtils();
    ui->ports_cbx->addItems(portUtils->getAvailablePortsNames());

    // Create instance for COM port utilities.
    this->comPortUtils = new ComPortUtils();

    // Disable 'stop' button by default.
    this->ui->stop_btn->setEnabled(false);

    // Set timer settings.
    this->oneSecTim = new QTimer(this);
    connect(oneSecTim, SIGNAL(timeout()), this, SLOT(timEvent()));
}

MainWindow::~MainWindow()
{
    delete oneSecTim;
    delete comPort;
    delete comPortUtils;
    delete ui;
}

void MainWindow::spdValueUpdated()
{    
    // Get raw data and process.
    QString rawDataBuffer = comPort->readData(2); // reversed 16bit number in hex

    // Get all the values needed.
    this->odomImpPerSec = comPortUtils->processRawData(rawDataBuffer); // int16_t

    float odomSpdPerSec_mms = ( ( odomImpPerSec * circleLen ) / DEFAULT_ODOM_IMP ) / ODOM_RATE; // mm/s
    float odomSpdPerSec_ms = odomSpdPerSec_mms * 3600 / 1000000;
    float odomSpdPerSec_kmh = odomSpdPerSec_mms / 1000;

    float odomAnglVelPerSec = ( ( odomImpPerSec * 2 * M_PI ) / DEFAULT_ODOM_IMP ) / ODOM_RATE; // rad/s

    this->ticksCounter++;

    // Set UI data labels with calculated values.
    this->ui->curimp_lbl->setText(QString::number(odomImpPerSec));
    this->ui->curspdKMH_lbl->setText(QString::number(odomSpdPerSec_kmh, 'f', 4));
    this->ui->curspdMS_lbl->setText(QString::number(odomSpdPerSec_ms, 'f', 2));
    this->ui->curanglvel_lbl->setText(QString::number(odomAnglVelPerSec, 'f', 2));
    this->ui->totaldstMS_lbl->setText(QString::number(totalDst_m, 'f', 2));
    this->ui->totaldstKMH_lbl->setText(QString::number(totalDst_m / 1000, 'f', 4));

    // Set raw data UI labels.
    this->ui->rawdata_lbl->setText(rawDataBuffer);
    this->ui->rawdata_int_lbl->setText(QString::number(odomImpPerSec));

    // Clear USART buffer.
    this->comPort->clearPortBuffer();
}

void MainWindow::timEvent()
{
    // todo: absolute value?
    this->totalDst_m += this->impValue * this->odomImpPerSec / 1000; // m

    this->ui->ticks_lbl->setText(QString::number(ticksCounter));
    ticksCounter = 0;
}

void MainWindow::on_run_btn_clicked()
{
    // Save default constants: wheelDiameter and impulseValue.
    this->circleLen = M_PI * this->ui->diameter_spnbx->value();
    this->impValue = this->circleLen / this->DEFAULT_ODOM_IMP;
    this->totalDst_m = 0;

    // Create and set up COM port instance.
    this->comPort = new ComPort(this->ui->ports_cbx->currentText());
    bool portOpened = comPort->openPort();
    if(!portOpened)
    {
        QMessageBox::warning(
                    this, "Port error",
                    tr("Cannot open a %1 port.\n").arg(this->ui->ports_cbx->currentText()),
                    QMessageBox::Yes);
        return;
    }

    comPort->clearPortBuffer();
    connect(comPort->getSerialPort(), SIGNAL(readyRead()),
            this, SLOT(spdValueUpdated()) );

    // Disable 'run' button.
    this->ui->run_btn->setEnabled(false);
    this->ui->stop_btn->setEnabled(true);

    // Reset and start timer.
    this->oneSecTim->start(1000);
}

void MainWindow::on_stop_btn_clicked()
{
    comPort->closePort();
    disconnect(comPort->getSerialPort(), SIGNAL(readyRead()),
                   this, SLOT(spdValueUpdated()) );

    // Disable 'stop' button.
    this->ui->run_btn->setEnabled(true);
    this->ui->stop_btn->setEnabled(false);

    // Stop one second timer.
    this->oneSecTim->stop();
}

void MainWindow::on_resetdst_btn_clicked()
{
    this->totalDst_m = 0;
    this->ui->totaldstMS_lbl->setText(QString::number(totalDst_m, 'f', 2));
    this->ui->totaldstKMH_lbl->setText(QString::number(totalDst_m / 1000, 'f', 4));
}

void MainWindow::on_MainWindow_destroyed()
{
    QApplication::quit();
}

void MainWindow::on_diameter_spnbx_valueChanged(int diameter)
{
    // default constants.
    this->circleLen = M_PI * diameter;
    this->impValue = this->circleLen / this->DEFAULT_ODOM_IMP;

    // update impulse value label.
    this->ui->impval_spnbx->setValue(impValue);
}
