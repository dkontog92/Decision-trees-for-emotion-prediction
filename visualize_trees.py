import matplotlib.pyplot as plt

def draw_subtree(tree_, ax, xc=0., yc=0., dx=10, dy=1, depth=0, from_=''):

    bbox_props = dict(boxstyle="round", fc="w", ec="0.7", alpha=0.5)

    # there is a node that has neither label nor kids
    if (len(tree_.label) is 0) and (0 not in tree_.kids):
        print(tree_.op)
        print(tree_.label)
        print(tree_.kids)

    if len(tree_.label) is not 0:
        if from_ == '':
            ha = 'center'
            va = 'center'
        elif from_ == '0':
            ha = 'right'
            va = 'top'
        elif from_ == '1':
            ha = 'left'
            va = 'top'
        print('label of %s' % tree_.op)
        ax.text(xc, yc, tree_.label, ha=ha, va=va, size=5,
                bbox=bbox_props)

    else:

        ax.text(xc, yc, tree_.op, ha="center", va="center", size=5,
                bbox=bbox_props)

        ax.arrow(x=xc, y=yc, dx=-dx, dy=-dy, width=0.02, lw=0, color='k', label='0')
        #ax.text(xc-dx/2, yc-dy, "0", ha="right", va="top", size=5)

        ax.arrow(x=xc, y=yc, dx=dx, dy=-dy, width=0.02, lw=0, color='k', label='0')
        #ax.text(xc+dx/2, yc-dy, "1", ha="left", va="top", size=5)

        # there is a node that has neither label nor kids
        try:
            print('kids of %s' % tree_.op)
            draw_subtree(tree_.kids[0], ax, xc=xc-dx, yc=yc-dy, dx=0.5*dx, dy=1.20*dy, depth=depth+1, from_='0')
            draw_subtree(tree_.kids[1], ax, xc=xc+dx, yc=yc-dy, dx=0.5*dx, dy=1.25*dy, depth=depth+1, from_='1')
        except:
            return tree_


def draw_tree(tree_, title=None):
    fig = plt.figure(1, figsize=(100,10))
    fig.clf()

    ax = fig.add_subplot(111)
    #ax.set_aspect(0.2)

    if title is not None:
        plt.title(title)

    draw_subtree(tree_, ax, xc=0., yc=0., dx=15, dy=1)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-60, 1)

    plt.draw()
    plt.show()
