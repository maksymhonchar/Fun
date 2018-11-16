#ifndef SORTDIALOG_H
#define SORTDIALOG_H

#include <QDialog>
#include "ui_dialog.h"

class SortDialog : public QDialog, public Ui::SortDialog
{
    Q_OBJECT

public:
    SortDialog(QWidget *parent = 0);
    void setColumnRange(QChar first, QChar last);

private slots:
    void on_moreButton_toggled(bool checked);
};

#endif // SORTDIALOG_H
